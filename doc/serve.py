#!/usr/bin/env python3
"""Simple HTTP server that serves files under a virtual base path."""

import http.server
import socketserver
import argparse
import os


class VirtualPathHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that requires and strips a base path prefix from requests."""
    
    base_path = ""
    
    def translate_path(self, path):
        # If base_path is set, require it as prefix
        if self.base_path:
            if path.startswith(self.base_path):
                path = path[len(self.base_path):] or "/"
            else:
                # Return a non-existent path to trigger 404
                return "/nonexistent-path-for-404"
        return super().translate_path(path)


class ReusableTCPServer(socketserver.TCPServer):
    """TCP server that allows port reuse."""
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Serve files under a virtual base path")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--base-path", "-b", default="", help="Virtual base path (e.g., /sionna)")
    parser.add_argument("--directory", "-d", default=".", help="Directory to serve")
    args = parser.parse_args()
    
    os.chdir(args.directory)
    VirtualPathHandler.base_path = args.base_path
    
    with ReusableTCPServer(("", args.port), VirtualPathHandler) as httpd:
        url = f"http://localhost:{args.port}{args.base_path}/"
        print(f"Serving {args.directory} at {url}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()
