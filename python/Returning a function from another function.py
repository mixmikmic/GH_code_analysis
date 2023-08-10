preheat_oven = lambda: print('Preheating oven')
put_croissants_in = lambda: print('Putting croissants in')
wait_five_minutes = lambda: print('Waiting five minutes')
take_croissants_out = lambda: print('Take croissants out (and eat them!)')

preheat_oven()
put_croissants_in()
wait_five_minutes()
take_croissants_out()

def perform_steps(*functions):
    
    for function in functions:
        function()
        
        
perform_steps(preheat_oven,
    put_croissants_in,
    wait_five_minutes,
    take_croissants_out)

def create_recipe(*functions):
    
    def run_all():
        
        for function in functions:
            function()
            
    return run_all


recipe = create_recipe(preheat_oven,
    put_croissants_in,
    wait_five_minutes,
    take_croissants_out)
recipe()

