# utils/subtitles.py
# simple subtitle parser wrapper using pysubs2 if available, fallback to basic SRT parse
try:
    import pysubs2
except:
    pysubs2 = None

class SubtitleParser:
    def __init__(self, path):
        self.path = path
        self.lines = []
        if pysubs2:
            try:
                subs = pysubs2.load(path)
                for ev in subs:
                    self.lines.append((ev.start/1000.0, ev.end/1000.0, ev.text))
            except:
                self._basic_load(path)
        else:
            self._basic_load(path)
    def _basic_load(self, path):
        # very basic SRT parsing
        with open(path, encoding='utf-8', errors='ignore') as f:
            content = f.read()
        blocks = content.strip().split('\\n\\n')
        for b in blocks:
            lines = b.split('\\n')
            if len(lines) >= 3:
                times = lines[1]
                start,end = times.split(' --> ')
                def t2s(s):
                    p = s.replace(',',':').split(':')
                    return int(p[0])*3600 + int(p[1])*60 + int(p[2]) + float("0."+p[3]) if len(p)>3 else 0
                try:
                    s = start.strip(); e = end.strip()
                    # naive convert
                    self.lines.append((0,999999, ' '.join(lines[2:])))
                except:
                    pass
    def get_text(self, current_sec):
        for s,e,t in self.lines:
            if s <= current_sec <= e:
                return t
        return ""