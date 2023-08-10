get_ipython().run_cell_magic('javascript', '', '\n// TASK:  Create an object\n\n\n\n// TASK save one attribute to the result variable (to print out)\nvar result = _____\n\n// A trick save these variables back to python variables to work with later\nIPython.notebook.kernel.execute(\'result="\' + result + \'";\');')

# This is python - just using it to print
print(result)

get_ipython().run_cell_magic('javascript', '', '\n// assign an anonymous func to a variable\nvar printBacon = function() {\n\n  // TASK:  return something, otherwise undefined\n};\n\n// TASK:  use this fucntion and place result in "result"\n\n\n// Our trick:  save these variables back to python variables to work with later\nIPython.notebook.kernel.execute(\'result="\' + result + \'";\');')

# In python
print(result)

get_ipython().run_cell_magic('javascript', '', '\nfunction Container(param) {\n    // this attaches "member" to an instance of this class\n    this.member = param;\n}\n\nvar myContainer = new Container(\'abc\');\n\n// TASK:  Access the string value in the instatiation of myContainer\n\n\nIPython.notebook.kernel.execute(\'result="\' + result + \'";\');')

print(result)

get_ipython().run_cell_magic('javascript', '', '\nfunction GymMember(name) {\n    this.membername = name;\n    \n    // These are private\n    var id = 3;\n    var that = this;\n    \n\n};\n\n\n// TASKS:  \n//    1) instance of class \n//    2) access id from the instance object\n//    3) write an anonymous function in class \n//    4) access that class method\n\n\n// What is the issue?\n\n// You can find more info at:  http://www.crockford.com/javascript/private.html\n\nIPython.notebook.kernel.execute(\'result="\' + result + \'";\');')

print(result)

get_ipython().run_cell_magic('javascript', '', '\n// a User class:\nfunction User() {\n    this.name = "";\n    this.life = 100;\n    this.giveLife = function giveLife(targetPlayer) {\n        targetPlayer.life += 1;\n        \n        // TASK:  We can give life points, but what about taking away life points\n    }\n}\n\n// Use class method\nvar Alfred = new User();\nvar Mich = new User();\n\n// Names were blank so give them name values\nAlfred.name = "Alfred";\nMich.name = "Mich";\n\n\n// TASK:  give the life points from Alfred to Mich\n\n\n// Save these variables back to python variables to work with later\nIPython.notebook.kernel.execute(\'Alfred_life="\' + Alfred.life + \'";\');\nIPython.notebook.kernel.execute(\'Mich_life="\' + Mich.life + \'";\');')

print(Alfred_life)
print(Mich_life)

get_ipython().run_cell_magic('javascript', '', '\n// a User class:\nfunction User() {\n    this.name = "";\n    this.life = 100;\n    this.giveLife = function giveLife(targetPlayer) {\n        targetPlayer.life += 1;        \n        this.life-=1;\n    }\n}\n\n// TASK:  Create a prototyped extra method to do something harmful to a player (e.g. punch)\n\n\n\n// Use class method\nvar Alfred = new User();\nvar Mich = new User();\n\n// Names were blank so give them name values\nAlfred.name = "Alfred";\nMich.name = "Mich";\n\n\n// give the life points from Alfred to Mich\nAlfred.giveLife(Mich);\n\n\n// TASK:  Use your harmful method\nAlfred.punch(Mich);\n\n\n// Save these variables back to python variables to work with later\nIPython.notebook.kernel.execute(\'Alfred_life="\' + Alfred.life + \'";\');\nIPython.notebook.kernel.execute(\'Mich_life="\' + Mich.life + \'";\');')

print(Alfred_life)
print(Mich_life)

get_ipython().run_cell_magic('javascript', '', '\n//==========IGNORE THIS STUFF============\n\nfunction sleep(milliseconds) {\n  var start = new Date().getTime();\n  for (var i = 0; i < 1e7; i++) {\n    if ((new Date().getTime() - start) > milliseconds){\n      break;\n    }\n  }\n}\n//========================================\n\n\n\nvar result;\nvar message = \'\'; // you will be modifying this\n\nfunction placeAnOrder(orderNumber) {\n    result = orderNumber;\n    \n    // It\'s a function as an argument\n    cookAndDeliverFood(function() {\n        // TASK: change the "message" var to reflect the order number (Hint: orderNumber.toString())\n    });\n\n}\n\n// This function we want to run asynchronously (but we\'re not here)\n// This is (should be) simulating some time consuming operation like connecting to a database\nfunction cookAndDeliverFood(callback) {\n    sleep(5000);\n    callback(); // usually this is done asynchronously with setTimeout\n    message += \' cooking, \';\n}\n\n// TASK:  Place a few orders\n\n\n\nIPython.notebook.kernel.execute(\'result="\' + message + \'";\');')

print(result)

get_ipython().run_cell_magic('javascript', '', '\nvar message = "";\n// var currentdate = new Date();\nvar currentdate = Date.now();\n// var currentsecs = currentdate.getSeconds()\nsetInterval(function() {\n    message += "hot dog \\n";\n}, 2000);\n\nvar nextdate = Date.now();\n\nvar result = currentdate.toString();\nvar result2 = nextdate.toString()\n\nIPython.notebook.kernel.execute(\'result="\' + result + \'";\');\nIPython.notebook.kernel.execute(\'result2="\' + result2 + \'";\');\nIPython.notebook.kernel.execute(\'result3="\' + message + \'";\');')

print(result)
print(result2)
print(result3)



