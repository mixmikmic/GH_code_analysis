get_ipython().run_cell_magic('capture', '', '%load_ext autoreload\n%autoreload 2\n%matplotlib inline\n# preamble\nimport sys\nsys.path.append("..")\nimport statnlpbook.transition as transition')

tokens = ["Economic", "news", "had", "little", "effect", "on", "financial", "markets", "."]
arcs = set([(1,0,"amod"),(2,1,"nsubj"), (2, 4, "dobj"), (4,3,"amod"), (4,5, "prep"), (5,7,"pmod"), (7,6,"amod")])

#transition.render_tree(tokens, arcs)

transition.render_displacy(*transition.to_displacy_graph(arcs, tokens))

tokens = ["ROOT", "Economic", "news", "had", "little", "effect", "on", "financial", "markets", "."]
arcs = set([(0,3, "root"), (0,9,"p"), (2,1,"amod"),(3,2,"nsubj"), (3, 5, "dobj"), (5,4,"amod"), (5,6, "prep"), (6,8,"pmod"), (8,7,"amod")])

transition.render_displacy(*transition.to_displacy_graph(arcs, tokens))

from collections import deque

class Configuration():
    def __init__(self, tokenized_sentence):
        # This implements the initial configuration for a sentence
        self.arcs = set()
        self.buffer = deque()
        self.sentence = tokenized_sentence
        for idx, token in enumerate(tokenized_sentence):
            self.buffer.append(token+ "_" + str(idx))
        self.stack = []
        
import copy
def parse(tokenized_sentence, actions):
    # This stores the (configuration, action) tuples generated
    transitions = []
    
    # Initialize the configuration
    configuration = Configuration(tokenized_sentence)
    transitions.append((copy.deepcopy(configuration), "INIT"))
    
    for action in actions:
        if action == "shift":
            token = configuration.buffer.popleft()
            configuration.stack.append(token)
        elif action == "reduce":
            # check if it is headed already:
            headed = False
            for arc in configuration.arcs:
                if arc[1] == int(dependentTokenId):
                    headed = True
            if not headed:
                raise Exception("Token at the top of the stack does not have an incoming edge.")
            
            configuration.stack.pop()
        elif action.startswith("leftArc"):
            # Get the dependent token
            dependentToken, dependentTokenId = configuration.stack.pop().split("_")

            # check if it is headed already:
            headed = False
            for arc in configuration.arcs:
                if arc[1] == int(dependentTokenId):
                    headed = True
            if headed:
                raise Exception("Dependent token has an incoming edge already")
            
            label = action.split("-")[1]
            headToken, headTokenId = configuration.buffer[0].split("_")
            
            configuration.arcs.add((int(headTokenId),int(dependentTokenId),label))

        elif action.startswith("rightArc"):
            label = action.split("-")[1]            
            
            dependent = configuration.buffer.popleft()
            dependentToken, dependentTokenId = dependent.split("_")
            headToken, headTokenId = configuration.stack[-1].split("_")
            
            configuration.arcs.add((int(headTokenId),int(dependentTokenId),label))
            
            configuration.stack.append(dependent)
            
        
        transitions.append((copy.deepcopy(configuration), action))
    
    if len(configuration.buffer) == 0:
        transitions.append((copy.deepcopy(configuration), "TERMINAL"))
    return transitions

tokenized_sentence = ["ROOT", "Economic", "news", "had", "little", "effect", "on", "financial", "markets", "."]
actions = ["shift","shift", "leftArc-amod", "shift", "leftArc-nsubj", "rightArc-root", "shift", "leftArc-amod", "rightArc-dobj", "rightArc-prep", "shift", "leftArc-amod", "rightArc-pmod", "reduce", "reduce", "reduce", "reduce", "rightArc-p"]

transitions = parse(tokenized_sentence, actions)

transition.render_transitions_displacy(transitions, tokenized_sentence)

tokens = ["ROOT", "What", "did", "economic", "news", "have", "little", "effect", "?"]
arcs = set([(0,5, "root"), (0,9,"p"), (8,1,"pobj"), (5,2,"aux"), (4,3,"amod"),(5,4,"nsubj"), (5, 7, "dobj"), (7,6,"amod"), (5,8, "prep"), (6,8,"pmod"), (8,7,"amod")])

transition.render_displacy(*transition.to_displacy_graph(arcs, tokens))

sentence = ["He", "wrote", "her", "a", "letter"]
actions1 = ["shift", "leftArc-nsubj", "shift", "rightArc-iobj", "reduce", "shift", "leftArc-det", "rightArc-dobj"]

transitions1 = parse(sentence, actions1)

transition.render_transitions_displacy(transitions1, sentence)

sentence = ["He", "wrote", "her", "a", "letter"]
actions2 = ["shift", "leftArc-nsubj", "shift", "rightArc-iobj", "shift", "leftArc-det", "reduce", "rightArc-dobj"]

transitions2 = parse(sentence, actions2)

transition.render_transitions_displacy(transitions2, sentence)



