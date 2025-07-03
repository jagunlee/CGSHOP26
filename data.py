class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y

class Diag:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    
    def intersect(self, d1):
        p1 = self.p1
        p2 = self.p2
        p3 = d1.p1
        p4 = d1.p2
        x1 = ((p2.y-p1.y)*(p4.x-p3.x)*p1.x-(p2.x-p1.x)*(p4.x-p3.x)*p1.y-(p4.y-p3.y)*(p2.x-p1.x)*p3.x+(p4.x-p3.x)*(p2.x-p1.x)*p3.y)/((p2.y-p1.y)*(p4.x-p3.x)-(p4.y-p3.y)*(p2.x-p1.x))
        if p1.x<x1 and x1<p2.x:
            return True
        return False

class Data:
    def __init__(self, pts, polygon=True):
        self.pts = pts



def turn(p1:Point, p2:Point, p3:Point):
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)

p1 = Point(1,1)
p2 = Point(2,2)
p3 = Point(1,2)
p4 = Point(2,1)

d1 = Diag(p1, p2)
d2 = Diag(p3, p4)
d3 = Diag(p1, p3)
print(d1.intersect(d2))
print(d1.intersect(d3))