"""A launched tool server answers loopback calls without Nagle stalls.

Before the tool server created its listener with IPPROTO_TCP and TCP_NODELAY,
asyncio left Nagle's algorithm on for accepted connections, and every MCP
response (headers, then body) waited for the client's delayed ACK: about 40 ms
per tool call on Linux, against about 3 ms without the stall.
"""

import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CALL = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {"name": "back", "arguments": {"message": "hi"}},
}
HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
    "mcp-protocol-version": "2025-06-18",
}


@pytest.mark.skipif(sys.platform != "linux", reason="delayed-ACK timing is Linux's")
def test_tool_calls_do_not_wait_for_delayed_acks(tmp_path):
    port_file = tmp_path / "port"
    env = {
        **os.environ,
        "MCP_PORT_FILE": str(port_file),
        "PYTHONPATH": os.pathsep.join(
            [str(FIXTURES), os.environ.get("PYTHONPATH", "")]
        ),
        "VF_STATE_URL": "",
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "echo_tool_v1"], env=env, start_new_session=True
    )
    try:
        deadline = time.monotonic() + 60
        while not port_file.exists() or not port_file.read_text().strip():
            assert server.poll() is None, "tool server exited"
            assert time.monotonic() < deadline, "tool server did not report a port"
            time.sleep(0.05)
        url = f"http://127.0.0.1:{port_file.read_text().strip()}/mcp"
        with httpx.Client(timeout=10) as client:
            while True:
                try:
                    client.post(url, json=CALL, headers=HEADERS)
                    break
                except httpx.TransportError:
                    assert time.monotonic() < deadline, "tool server not serving"
                    time.sleep(0.05)
            timings = []
            for _ in range(15):
                start = time.perf_counter()
                response = client.post(url, json=CALL, headers=HEADERS)
                timings.append(time.perf_counter() - start)
                assert response.status_code == 200
                assert "ok-7f3" in response.text
        assert statistics.median(timings) < 0.02, timings
    finally:
        server.terminate()
        server.wait(timeout=10)
