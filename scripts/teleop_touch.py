"""A phone controller that works on ANY phone. No WebXR, no ARCore, no cert.

    .venv/Scripts/python scripts/teleop_touch.py

Then open the printed http:// URL on the phone and drag.

WHY THIS EXISTS. The `teleop` package's page calls
`navigator.xr.requestSession('immersive-ar')`, which needs Android Chrome with
Google Play Services for AR. On iOS Safari WebXR does not exist at all: the
page loads, the websocket connects, and it then streams nothing, which from the
server looks identical to a network problem. Measured exactly that -- "Client
connected" followed by zero pose samples.

So this serves a plain touch surface over plain HTTP instead:

    drag anywhere on the pad -> X / Y, accumulated like a trackpad
    drag the Z strip         -> height
    ENGAGE (a toggle)        -> arm follows; tap again to stop
    RECENTRE                 -> zero the accumulator without moving the arm
    the grip slider          -> jaw opening

Drag is CUMULATIVE, not absolute. With an absolute pad, lifting a finger and
putting it down elsewhere teleports the target, which either jerks the arm or
trips the dropout guard. Accumulating deltas lets you stroke repeatedly to
travel further than the pad is wide.

It is 3-DoF translation plus a grip, not a 6-DoF pose, and it is not pretending
otherwise -- orientation is left at identity and `teleop_sim` holds the gripper
pointing down. What it IS, is the same wire format, the same port, and the same
`labgen.devices.PoseSource` slot as the WebXR bridge and the Quest. The sim
cannot tell which one is driving it.

Stdlib only, so it runs in the project venv with nothing installed. Plain HTTP
on purpose: no WebXR means no secure-context requirement, which removes the
self-signed certificate warning and the whole class of failures around it.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_BRIDGE_PORT = 9871
DEFAULT_WEB_PORT = 8000

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>labgen teleop</title>
<style>
  :root { --bg:#12171a; --fg:#e6eeec; --dim:#7d8f8b; --line:#2b3a38;
          --accent:#3fb6a8; --live:#d97b28; }
  * { box-sizing:border-box; -webkit-user-select:none; user-select:none;
      -webkit-touch-callout:none; -webkit-tap-highlight-color:transparent; }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
              font:15px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
              overscroll-behavior:none; touch-action:none; }
  .wrap { display:flex; flex-direction:column; height:100%; padding:8px; gap:8px; }
  .row { display:flex; gap:8px; align-items:stretch; }
  #view { width:100%; border-radius:10px; border:1px solid var(--line);
          background:#0d1113; aspect-ratio:16/9; object-fit:contain; }
  #pad { flex:1; position:relative; border:1px solid var(--line); border-radius:10px;
         background:
           repeating-linear-gradient(0deg,#1a2124 0 1px,transparent 1px 34px),
           repeating-linear-gradient(90deg,#1a2124 0 1px,transparent 1px 34px), #151c1f; }
  #pad.on { border-color:var(--live); }
  #pad .lbl { position:absolute; top:8px; left:10px; color:var(--dim); font-size:11px;
              letter-spacing:.1em; text-transform:uppercase; }
  #zstrip { width:74px; position:relative; border:1px solid var(--line); border-radius:10px;
            background:repeating-linear-gradient(0deg,#1a2124 0 1px,transparent 1px 34px), #151c1f;
            display:flex; align-items:center; justify-content:center;
            color:var(--dim); font-size:11px; letter-spacing:.1em; }
  #zstrip.on { border-color:var(--live); }
  .btn { border:1px solid var(--line); border-radius:10px; background:#243033;
         color:var(--fg); font-size:17px; font-weight:600; letter-spacing:.03em;
         height:62px; flex:1; }
  .btn.live { background:var(--live); border-color:var(--live); color:#120d07; }
  .btn.sm { flex:1; font-size:14px; font-weight:500; }
  .btn.on { background:var(--accent); border-color:var(--accent); color:#06201e; }
  #grip { flex:1; accent-color:var(--accent); height:44px; }
  #status { color:var(--dim); font-size:12px; font-variant-numeric:tabular-nums;
            display:flex; justify-content:space-between; gap:10px; }
  #status b { color:var(--fg); font-weight:600; }
</style></head>
<body><div class="wrap">
  <div id="status"><span id="pos">x 0  y 0  z 0 mm</span><span id="rate">--</span></div>
  <img id="view" src="/stream" alt="sim view">
  <div class="row" style="flex:1">
    <div id="pad"><div class="lbl">drag anywhere &middot; x / y</div></div>
    <div id="zstrip">Z</div>
  </div>
  <div class="row">
    <button id="armL" class="btn sm">LEFT</button>
    <button id="armR" class="btn sm on">RIGHT</button>
    <button id="rec" class="btn sm">RECENTRE</button>
  </div>
  <div class="row"><input id="grip" type="range" min="0" max="100" value="0"></div>
  <div class="row"><button id="go" class="btn">ENGAGE</button></div>
  <div class="row">
    <button id="save" class="btn sm">SAVE DEMO</button>
    <button id="discard" class="btn sm">DISCARD</button>
  </div>
</div>
<script>
const pad=document.getElementById('pad'), zs=document.getElementById('zstrip');
const go=document.getElementById('go'), rec=document.getElementById('rec');
const grip=document.getElementById('grip'), rate=document.getElementById('rate');
const posEl=document.getElementById('pos');
const armL=document.getElementById('armL'), armR=document.getElementById('armR');

// Which arm this device drives. Two phones can open this page and pick one
// each; the sim routes on the `hand` field and neither device knows the other
// exists. A single phone can also switch, which is how one person tests a
// bimanual scene alone.
let hand = new URLSearchParams(location.search).get('hand') || 'right';   // right arm = robot0, the one the vial task uses
function setHand(h){
  hand = h;
  armL.classList.toggle('on', h === 'left');
  armR.classList.toggle('on', h === 'right');
  // Switching arms must not carry the previous arm's accumulated offset over,
  // or the new arm lurches by however far the old one had travelled.
  X = Y = Z = 0;
  setEngaged(false);
}

// Cumulative, like a trackpad -- NOT the absolute position of the finger in the
// pad. With an absolute pad, lifting your finger and putting it down somewhere
// else teleports the target, which either jerks the arm or trips the dropout
// guard. Accumulating deltas means you can stroke repeatedly to travel further
// than the pad is wide, exactly like a mouse on a small desk.
let X=0, Y=0, Z=0, engaged=false, sent=0, fails=0;
// Episode control. Sent on every sample with a counter, so the sim -- which
// only reads the newest sample -- cannot miss a press. It acts on a change.
let cmd='', cmdSeq=0;
const MM_PER_PX = 0.6;           // a 300 px stroke moves the hand 180 mm

function dragger(el, apply){
  let lx=null, ly=null;
  const down = e => { const t=e.touches?e.touches[0]:e; lx=t.clientX; ly=t.clientY;
                      el.classList.add('on'); e.preventDefault(); };
  const move = e => { if(lx===null) return; const t=e.touches?e.touches[0]:e;
                      apply(t.clientX-lx, t.clientY-ly); lx=t.clientX; ly=t.clientY;
                      e.preventDefault(); };
  const up   = e => { lx=ly=null; el.classList.remove('on'); };
  el.addEventListener('touchstart',down,{passive:false});
  el.addEventListener('touchmove',move,{passive:false});
  el.addEventListener('touchend',up); el.addEventListener('touchcancel',up);
  el.addEventListener('mousedown',down);
  el.addEventListener('mousemove',e=>{ if(e.buttons) move(e); });
  window.addEventListener('mouseup',up);
}
dragger(pad,(dx,dy)=>{ X += dx*MM_PER_PX/1000; Y -= dy*MM_PER_PX/1000; });
dragger(zs ,(dx,dy)=>{ Z -= dy*MM_PER_PX/1000; });

function setEngaged(v){
  engaged=v; go.classList.toggle('live',v); go.textContent=v?'ENGAGED — TAP TO STOP':'ENGAGE';
}
// A TOGGLE, not hold-to-move. Holding a button while dragging with the same
// hand is the thing that made this unusable.
go.addEventListener('click',()=>setEngaged(!engaged));
armL.addEventListener('click',()=>setHand('left'));
armR.addEventListener('click',()=>setHand('right'));
rec.addEventListener('click',()=>{ X=Y=Z=0; });
function episode(c){ cmd=c; cmdSeq++; X=Y=Z=0; setEngaged(false); }
document.getElementById('save').addEventListener('click',()=>episode('save'));
document.getElementById('discard').addEventListener('click',()=>episode('discard'));
// Still auto-stops if the page goes away: an arm that keeps moving because the
// phone locked is the worst failure available.
document.addEventListener('visibilitychange',()=>{ if(document.hidden) setEngaged(false); });

async function tick(){
  posEl.textContent=hand.toUpperCase()+'  x '+(X*1000).toFixed(0)
                    +'  y '+(Y*1000).toFixed(0)+'  z '+(Z*1000).toFixed(0)+' mm';
  const body=JSON.stringify({
    position:{x:X, y:Y, z:Z}, orientation:{w:1,x:0,y:0,z:0},
    move:engaged, gripper:(+grip.value)/100.0, scale:1.0,
    seq:++sent, hand:hand, cmd:cmd, cmd_seq:cmdSeq });
  try{ await fetch('/pose',{method:'POST',body,
       headers:{'Content-Type':'application/json'},keepalive:true}); }
  catch(err){ fails++; }
  rate.innerHTML='<b>'+sent+'</b> sent'+(fails?(' / '+fails+' failed'):'');
}
setInterval(tick, 33);
</script></body></html>
"""


class Fanout:
    """Holds sim clients, drops the ones that go away. Never blocks the phone."""

    def __init__(self) -> None:
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self.dropped = 0
        self.stalled = 0

    def add(self, sock: socket.socket) -> None:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Blocking with a short timeout, NOT non-blocking. sendall on a
        # non-blocking socket raises BlockingIOError the moment the kernel
        # buffer is momentarily full, and BlockingIOError IS an OSError --
        # so the 'drop dead clients' branch deleted the live sim on the
        # first hiccup. It showed up as the client count going 1 -> 0 and
        # the arm quietly ignoring the phone.
        sock.settimeout(0.25)
        with self._lock:
            self._clients.append(sock)

    def broadcast(self, line: bytes) -> None:
        with self._lock:
            for c in list(self._clients):
                try:
                    c.sendall(line)
                except (BlockingIOError, socket.timeout):
                    # Momentary backpressure. Skip this sample; the next one is
                    # fresher anyway. Dropping the client here is what broke it.
                    self.stalled += 1
                except OSError:
                    self.dropped += 1
                    self._clients.remove(c)
                    try:
                        c.close()
                    except OSError:
                        pass

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._clients)


def local_ips() -> list[str]:
    out = []
    for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
        ip = info[4][0]
        if not ip.startswith(("127.", "169.254.")) and ip not in out:
            out.append(ip)
    return out


CRLF = bytes([13, 10])


class FrameStore:
    """Holds the most recent rendered frame. One slot, overwritten.

    A queue would be wrong: a viewer that falls behind should skip to now, not
    play back the past. Same reason the control loop drains to the newest pose.
    """

    def __init__(self) -> None:
        self._jpeg = None
        self._lock = threading.Lock()
        self._seq = 0
        self._new = threading.Condition(self._lock)

    def put(self, jpeg: bytes) -> None:
        with self._new:
            self._jpeg = jpeg
            self._seq += 1
            self._new.notify_all()

    def wait_for_next(self, last_seq: int, timeout: float = 5.0):
        with self._new:
            if self._seq == last_seq:
                self._new.wait(timeout)
            return self._jpeg, self._seq

    @property
    def seq(self) -> int:
        return self._seq


def serve_frame_producer(store: "FrameStore", port: int) -> None:
    """The sim connects OUT to here and pushes length-prefixed JPEGs.

    Outward from WSL again, for the same reason the pose stream runs that way:
    under default NAT the Windows side cannot open a connection into WSL, but
    WSL can always reach the Windows host.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(2)
    print(f"[touch] sim pushes frames here: 0.0.0.0:{port}", flush=True)
    while True:
        sock, addr = srv.accept()
        print(f"[touch] frame producer connected from {addr}", flush=True)
        buf = b""
        try:
            while True:
                while len(buf) < 4:
                    chunk = sock.recv(262144)
                    if not chunk:
                        raise ConnectionError
                    buf += chunk
                n = int.from_bytes(buf[:4], "big")
                buf = buf[4:]
                while len(buf) < n:
                    chunk = sock.recv(262144)
                    if not chunk:
                        raise ConnectionError
                    buf += chunk
                store.put(buf[:n])
                buf = buf[n:]
        except (ConnectionError, OSError):
            print("[touch] frame producer disconnected", flush=True)
        finally:
            try:
                sock.close()
            except OSError:
                pass


def make_handler(fan: Fanout, state: dict, store: "FrameStore"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):        # quiet; the loop prints its own
            pass

        def do_GET(self):
            if self.path == "/stream":
                self.do_stream()
                return
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_stream(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    jpeg, seq = store.wait_for_next(seq)
                    if jpeg is None:
                        continue
                    head = (b"--frame" + CRLF
                            + b"Content-Type: image/jpeg" + CRLF
                            + b"Content-Length: " + str(len(jpeg)).encode()
                            + CRLF + CRLF)
                    self.wfile.write(head + jpeg + CRLF)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def do_POST(self):
            if self.path != "/pose":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                state["malformed"] += 1
                return
            msg["t"] = time.time()
            msg["device"] = "touch"
            # The page picks the arm; the server only supplies a default
            # for a client that never said.
            msg.setdefault("hand", state["hand"])
            state["samples"] += 1
            fan.broadcast((json.dumps(msg) + "\n").encode("utf-8"))
    return Handler


def serve_sim_clients(fan: Fanout, port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(8)
    print(f"[touch] sim clients connect here: 0.0.0.0:{port}", flush=True)
    while True:
        sock, addr = srv.accept()
        print(f"[touch] sim connected from {addr}", flush=True)
        fan.add(sock)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    ap.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT)
    ap.add_argument("--hand", default="right")
    ap.add_argument("--frame-port", type=int, default=9872)
    args = ap.parse_args()

    fan = Fanout()
    store = FrameStore()
    state = {"samples": 0, "malformed": 0, "hand": args.hand}
    threading.Thread(target=serve_sim_clients, args=(fan, args.bridge_port),
                     daemon=True).start()
    threading.Thread(target=serve_frame_producer, args=(store, args.frame_port),
                     daemon=True).start()

    httpd = ThreadingHTTPServer(("0.0.0.0", args.web_port),
                                make_handler(fan, state, store))
    print("[touch] open ONE of these on the phone (plain http, no certificate):")
    for ip in local_ips():
        print(f"           http://{ip}:{args.web_port}")
    print("[touch] hold the big button and drag the pad.", flush=True)

    def report():
        last = 0
        while True:
            time.sleep(3.0)
            print(f"[touch] {state['samples']} samples "
                  f"(+{state['samples'] - last}), {fan.count} sim client(s), "
                  f"{state['malformed']} malformed", flush=True)
            last = state["samples"]
    threading.Thread(target=report, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
