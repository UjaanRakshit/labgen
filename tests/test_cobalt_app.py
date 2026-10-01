"""The COBALT app protocol, against COBALT's own semantics (device_state.py,
yam_robot.py), and the bridge end to end with a fake app speaking real
WebSocket frames."""

import base64
import json
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from labgen.cobalt_app import (PHONE_TO_ROBOT, POSITION_SCALE, DeviceState, read_frame,
                               rotation_to_wxyz, send_frame, to_robot)
from labgen.devices import PoseEvent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def msg(**kw):
    m = {"id": 1, "enable": 1, "dpos": [0.01, 0.0, 0.0], "grasp": 0, "reset": 0, "completion": 0,
         "timeout": 0, "valid": 1, "keep_demo_decision": 1, "demo_decision_indicator": 1,
         "timestamp": 0.0, "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    m.update(kw)
    return m


def test_reset_rising_edge_toggles_the_indicator_and_holds_control_until_acknowledged():
    d = DeviceState("a")
    s = d.parse(msg(reset=1))
    assert s.user_reset and json.loads(d.response())["reset"] == 1
    assert not s.engaged, "control is disabled during the reset handshake"
    assert not d.parse(msg(reset=1)).user_reset, "only the rising edge counts"
    assert d.parse(msg(reset=0)).engaged, "the app dropping reset ends the handshake"


def test_completion_stays_raised_until_the_app_toggles_its_indicator():
    d = DeviceState("a")
    d.signal_completion()
    d.parse(msg(completion=0))
    assert json.loads(d.response())["complete"] == 1
    d.parse(msg(completion=1))                     # the app handled it
    assert json.loads(d.response())["complete"] == 0


def test_cobalts_first_sample_guard():
    """Engaged and valid but dpos all zero and identity rotation: not a reading."""
    assert not DeviceState("a").parse(msg(dpos=[0.0, 0.0, 0.0])).valid


def test_cobalts_yam_mapping():
    dp, R = to_robot([0.03, 0.06, 0.09], np.eye(3))
    assert np.allclose(dp, [-0.02, -0.04, 0.06])  # diag(-1,-1,1) / 1.5
    assert np.allclose(R, PHONE_TO_ROBOT) and POSITION_SCALE == pytest.approx(1 / 1.5)


@pytest.mark.parametrize("R", [np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, -1.0, 1.0]),
                               np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])])
def test_rotation_to_wxyz_round_trips_including_half_turns(R):
    w, x, y, z = rotation_to_wxyz(R)
    ev = PoseEvent.from_json({"position": {"x": 0, "y": 0, "z": 0},
                              "orientation": {"w": w, "x": x, "y": y, "z": z}})
    assert np.allclose(ev.rotation, R, atol=1e-9)


def _client(port, path="/ws?config=%7B%7D"):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    key = base64.b64encode(b"0123456789abcdef").decode()
    s.sendall((f"GET {path} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(1024)
    assert b" 101 " in buf.split(b"\r\n")[0]
    return s


def test_bridge_end_to_end_with_a_fake_app():
    import cobalt_app_bridge as B
    from teleop_touch import Fanout

    sim_a, sim_b = socket.socketpair()
    fan = Fanout()
    fan.add(sim_a)
    session = B.Session(fan)
    session.sims = 1
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def accept():
        conn, addr = srv.accept()
        B.serve_app(conn, addr, session, "127.0.0.1")
    threading.Thread(target=accept, daemon=True).start()

    app = _client(port)
    op, init = read_frame(app)
    assert json.loads(init)["type"] == "init"
    op, status = read_frame(app)
    assert json.loads(status) == {"type": "status", "data": {"ready": True}}

    send_frame(app, json.dumps({"type": "device data", "data": msg(dpos=[0.03, 0.0, 0.0])}).encode(),
               mask=True)
    op, resp = read_frame(app)
    r = json.loads(resp)
    assert r["type"] == "response" and json.loads(r["data"]) == {"reset": 0, "complete": 0, "timeout": 0}

    sim_b.settimeout(5)
    line = sim_b.recv(65536).split(b"\n")[0]
    ev = PoseEvent.from_json(json.loads(line))
    assert ev.engaged and ev.hand == "right" and ev.device == "cobalt-app"
    assert np.allclose(ev.position, [-0.02, 0.0, 0.0])          # 0.03 phone x -> -x robot, /1.5
    assert np.allclose(ev.rotation, PHONE_TO_ROBOT)

    session.on_success()
    send_frame(app, json.dumps({"type": "device data", "data": msg()}).encode(), mask=True)
    op, resp = read_frame(app)
    assert json.loads(json.loads(resp)["data"])["complete"] == 1
    app.close()
    srv.close()
