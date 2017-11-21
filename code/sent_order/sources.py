

class Corpus:

    def __init__(self, path):
        self.path = path

    def lines(self):
        with open(self.path) as fh:
            for line in fh:
                yield line.strip()

    def abstract_lines(self):
        """Generate abstract line groups.
        """
        lines = []
        for line in self.lines():
            if line:
                lines.append(line)
            else:
                yield lines
                lines = []
