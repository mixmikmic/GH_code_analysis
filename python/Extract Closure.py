class Grade:
    def __init__(self, student, score):
        self.student = student
        self.score =  score


def print_stats(grades):
    if not grades:
        raise ValueError('Must supply at least one Grade')
        
    total, count = 0, 0
    low, high = float('inf'), float('-inf')
    for grade in grades:
        total += grade.score
        count += 1
        if grade.score < low:
            low = grade.score
        elif grade.score > high:
            high = grade.score

    average = total / count

    print('Average score: %.1f, low score: %.1f, high score %.1f' %
          (average, low, high))

grades = [Grade('Bob', 92), Grade('Sally', 89), Grade('Roger', 73), Grade('Alice', 96)]
print_stats(grades)

def print_stats(grades):
    if not grades:
        raise ValueError('Must supply at least one Grade')
        
    total, count = 0, 0
    low, high = float('inf'), float('-inf')

    def adjust_stats(grade):
        nonlocal total, count, low, high
        total += grade.score
        count += 1
        if grade.score < low:
            low = grade.score
        elif grade.score > high:
            high = grade.score

    for grade in grades:
        adjust_stats(grade)
            
    average = total / count

    print('Average score: %.1f, low score: %.1f, high score %.1f' %
          (average, low, high))

print_stats(grades)

class CalculateStats:
    def __init__(self):
        self.total = 0
        self.count = 0
        self.low = float('inf')
        self.high = float('-inf')

    def __call__(self, grades):
        for grade in grades:
            self.total += grade.score
            self.count += 1
            if grade.score < self.low:
                self.low = grade.score
            elif grade.score > self.high:
                self.high = grade.score

                
def print_stats(grades):
    if not grades:
        raise ValueError('Must supply at least one Grade')

    stats = CalculateStats()
    stats(grades)
    average = stats.total / stats.count

    print('Average score: %.1f, low score: %.1f, high score %.1f' %
          (average, stats.low, stats.high))

print_stats(grades)

class CalculateStats:
    def __init__(self):
        self.total = 0
        self.count = 0
        self.low = float('inf')
        self.high = float('-inf')

    def __call__(self, grades):
        for grade in grades:
            self.total += grade.score
            self.count += 1
            if grade.score < self.low:
                self.low = grade.score
            elif grade.score > self.high:
                self.high = grade.score

    @property
    def average(self):
        return self.total / self.count

    
def print_stats(grades):
    if not grades:
        raise ValueError('Must supply at least one Grade')

    stats = CalculateStats()
    stats(grades)

    print('Average score: %.1f, low score: %.1f, high score %.1f' %
          (stats.average, stats.low, stats.high))

print_stats(grades)

