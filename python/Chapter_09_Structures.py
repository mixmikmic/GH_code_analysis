# Program to display student details
# class declaration instead of a structure

class student:
    rno = 0
    sname = ""
    tot = 0

# class object variable

x = student()
x.rno = 20147
x.sname = "PRADEEP"
x.tot = 64

print "Enter roll number, name and total marks"
print x.rno,x.sname,x.tot
print "     Details entered are"
print "Roll No.       :",x.rno
print "Student name   :",x.sname
print "Total marks    :",x.tot
print "Press any key to continue. . ."

# Program to input student details and print the marks of a specified student as output

# class declaration for structure

class student:
    def __init__(self,rno,sname,tot):
        self.rno = rno
        self.sname = sname
        self.tot = tot
    
# variable declaration

ch = "y"
n = 3

# details of n students
std = []
std.append(student(20201,"ARUN",78))
std.append(student(20208,"DEEPAK",69))
std.append(student(20223,"SUSMITHA",88))

print "How many students ?",n
print "Roll number ?",std[0].rno
print "Name ?",std[0].sname
print "Total marks ?",std[0].tot
print "Roll number ?",std[1].rno
print "Name ?",std[1].sname
print "Total marks ?",std[1].tot
print "Roll number ?",std[2].rno
print "Name ?",std[2].sname
print "Total marks ?",std[2].tot
print 


# To display marks of the student
while ch == "y" or ch == "Y":
    temp = 20208
    print "Enter student roll number to display marks :",temp
    print 
    flag = 0
    #loop to search and display details
    for i in range(3):
        if flag == 0:
            if std[i].rno == temp:
                print "Marks obtained by ",std[i].rno,std[i].sname
                print "Total   :",std[i].tot
                flag = 1
    if flag == 0:
        print temp," is not present in the list "
        
    ch = "n"    
    print "press  y - to continue"
    print "       any other key to stop.",ch
    

# Program to declare a structure for student details and display list of students who obtained more than 75 marks

# class for student structure

class student:
    def __init__(self,rno,sname,tot):
        self.rno = rno
        self.sname = sname
        self.tot = tot

std = []
std.append(student(30401,"ANAND",59))
std.append(student(30404,"NIRMAL",64))
std.append(student(30428,"ISWARYA",82))
std.append(student(30432,"VIVEKA",79))

n = 4
print "How many students ?",n
print "Roll Number ?",std[0].rno
print "Name ?",std[0].sname
print "Total marks ?",std[0].tot
print "Roll Number ?",std[1].rno
print "Name ?",std[1].sname
print "Total marks ?",std[1].tot
print "Roll Number ?",std[2].rno
print "Name ?",std[2].sname
print "Total marks ?",std[2].tot
print "Roll Number ?",std[3].rno
print "Name ?",std[3].sname
print "Total marks ?",std[3].tot
print 


print "----------------------------------------------------"
print "  Roll No.           Name            Total marks    "
print "----------------------------------------------------"
for i in range(n):
    if std[i].tot >= 75:
        print "    ",std[i].rno,"          ",std[i].sname,"         ",std[i].tot

print "----------------------------------------------------"
        

# Program to store employee information and to compute employee's pay

import math
# class declaration for employee
class employee:
    def __init__(self,eno,ename,epay,jdate):
        self.eno = eno
        self.ename = ename
        self.epay = epay
        self.jdate = jdate

employs = []
employs.append(employee(20101,"ASHIKA",1000,"31/04/2001"))
employs.append(employee(20182,"ASHWIN",6000,"11/12/1995"))
employs.append(employee(20204,"PRAVEEN",3000,"18/06/1994"))

n = 3

print "Employee No. ?",employs[0].eno
print "Name ?",employs[0].ename
print "Existing date ?",employs[0].epay
print "Joinin date ?",employs[0].jdate
print 
print "Press y- to continue any other key to stop. y"
print "Employee No. ?",employs[1].eno
print "Name ?",employs[1].ename
print "Existing date ?",employs[1].epay
print "Joinin date ?",employs[1].jdate
print
print "Press y- to continue any other key to stop. y"
print "Employee No. ?",employs[2].eno
print "Name ?",employs[2].ename
print "Existing date ?",employs[2].epay
print "Joinin date ?",employs[2].jdate
print
print "Press y- to continue any other key to stop. N"
print
print n," records are entered"
print "Press any key to print the revised salary list"
print 



def revise(temp):
    if temp <= 2000:
        temp = int(temp + math.ceil(temp * 0.15))
        return temp
    elif temp <= 5000:
        temp = int(temp + temp * 0.10)
        return temp
    else:
        return temp

    
# loop to increment salary
for i in range(n):
    employs[i].epay = revise(employs[i].epay)

    


# loop to print revised salary list
print "                 Employees Revised Pay List          "
print "---------------------------------------------------------"
print " S.No.     Number      Name      Joining date      Pay   "
print "---------------------------------------------------------"

for i in range(n):
    print " ",i+1,"     ",employs[i].eno,"      ",employs[i].ename,"     ",employs[i].jdate,"     ",employs[i].epay

print "---------------------------------------------------------"    
    

# Program to store cricket details and to display a team-wise list with batting average

# class for cricket structure

class cricket:
    def __init__(self,pname,tname,bavg):
        self.pname = pname
        self.tname = tname
        self.bavg = bavg

n = 6
probable = []
probable.append(cricket("KUMBLE","KARNATAKA",22))
probable.append(cricket("KAMBLI","MUMBAI",39))
probable.append(cricket("SRIKANTH","TAMILNADU",52))
probable.append(cricket("SACHIM","MUMBAI",69))
probable.append(cricket("RAHUL","KARNATAKA",57))
probable.append(cricket("RAMESH","TAMILNADU",48))

print "How many players ?",n
print
print "Player name ?",probable[0].pname
print "Which team ?",probable[0].tname
print "Batting average ?",probable[0].bavg
print "Player name ?",probable[1].pname
print "Which team ?",probable[1].tname
print "Batting average ?",probable[1].bavg
print "Player name ?",probable[2].pname
print "Which team ?",probable[2].tname
print "Batting average ?",probable[2].bavg
print "Player name ?",probable[3].pname
print "Which team ?",probable[3].tname
print "Batting average ?",probable[3].bavg
print "Player name ?",probable[4].pname
print "Which team ?",probable[4].tname
print "Batting average ?",probable[4].bavg
print "Player name ?",probable[5].pname
print "Which team ?",probable[5].tname
print "Batting average ?",probable[5].bavg
print
print 
j = 0
teams = []
teams.append(probable[0].tname)
j = j + 1
for i in range(n):
    flag = 0
    for k in range(j):
        if flag == 0:
            if probable[i].tname == teams[k]:
                flag = 1
            if flag == 0 :
                teams.append(probable[i].tname)
                j = j + 1

# loop to print team-wise list

for k in range(3):
    print "            ",teams[k]
    print "---------------------------------------------"
    for i in range(n):
        if probable[i].tname == teams[k]:
            print "   ",probable[i].pname,"             ",probable[i].bavg
    print "---------------------------------------------"   





# Program to illustrate the use of union using integer,string and float

class student:
    def __init__(self,roll_no,sname,marks):
        self.roll_no = roll_no
        self.sname = sname
        self.marks = marks

std = []
std.append(student(0,"AJITH",0))
std.append(student(0,"RAJU",0))
std.append(student(0,"VIGNESH",0))
std.append(student(0,"DIVYA",0))

ch = 2
print "-----------------------------"
print "          Main menu          "
print "-----------------------------"
print "Press 1 to enter roll numbers"
print "      2 to enter names       "
print "      3 to enter marks       "
print "      4 to stop              "
print 
print "Enter your choice :",ch

n = 4
print "How many students?",n
for i in range(n):
    print "    Student name ? ",std[i].sname

print
print 
# display required list
# switch case
if ch == 1:  #case 1
    print "------------------------------"
    print "   Students roll number list  "
    print "------------------------------"
    for i in range(n):
        print std[i].roll_no
    print "-------------------------------"
elif ch == 2:  # case 2
    print "------------------------------"
    print "   Students name list  "
    print "------------------------------"
    for i in range(n):
        print std[i].sname
    print "-------------------------------"
elif ch == 3: # case 3
    print "------------------------------"
    print "   Students mark list  "
    print "------------------------------"
    for i in range(n):
        print "Student marks",std[i].roll_no
    print "-------------------------------"
    
    

