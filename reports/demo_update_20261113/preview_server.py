from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re, os
ROOT=Path(__file__).resolve().parent/'site/dist'
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
 def send_head(self):
  self._range=None;p=Path(self.translate_path(self.path));r=self.headers.get('Range')
  if not r or not p.is_file():return super().send_head()
  m=re.fullmatch(r'bytes=(\d+)-(\d*)',r)
  if not m:return super().send_head()
  n=p.stat().st_size;start=int(m[1]);end=min(int(m[2]) if m[2] else n-1,n-1)
  if start>=n:self.send_error(416);return None
  f=p.open('rb');f.seek(start);self._range=end-start+1
  self.send_response(206);self.send_header('Content-Type',self.guess_type(str(p)))
  self.send_header('Content-Length',str(self._range));self.send_header('Content-Range',f'bytes {start}-{end}/{n}')
  self.send_header('Accept-Ranges','bytes');self.end_headers();return f
 def copyfile(self,src,dst):
  if self._range is None:return super().copyfile(src,dst)
  left=self._range
  while left:
   b=src.read(min(left,65536))
   if not b:break
   dst.write(b);left-=len(b)
 def log_message(self,*args):pass
ThreadingHTTPServer(('127.0.0.1',8768),Handler).serve_forever()
