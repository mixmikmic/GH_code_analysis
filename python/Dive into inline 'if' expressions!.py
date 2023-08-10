def p_grade_description(gp):
    
    """Dutch grades range between 0 and 10"""
    
    if gp > 7:
        return 'good'
    if gp > 5:
        return 'sufficient'
    return 'insufficient'

p_grade_description(8)

(lambda gp: 'good' if gp > 7 else 'sufficient' if gp > 5 else 'insufficient')(6)

gender_code = 1
gender = 'female' if gender_code else 'male'
print(gender)

