class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __eq__(self, p):
        return self.x == p.x and self.y == p.y

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def __ne__(self, p):
        return self.x != p.x or self.y != p.y

    def __lt__(self, p):
        return (self.x, self.y) < (p.x, p.y)

    def __le__(self, p):
        return (self.x, self.y) <= (p.x, p.y)

    def __gt__(self, p):
        return (self.x, self.y) > (p.x, p.y)

    def __ge__(self, p):
        return (self.x, self.y) >= (p.x, p.y)