import pandas as pd
import matplotlib.pyplot as plt

poke_dict = {
 'Name': {1: 'Bulbasaur',
          2: 'Ivysaur',
          3: 'Venusaur',
          4: 'Charmander',
          5: 'Charmeleon',
          6: 'Charizard',
          7: 'Squirtle',
          8: 'Wartortle',
          9: 'Blastoise'},
 'Attack': {1: 49, 2: 62, 3: 82, 4: 52, 5: 64, 6: 84, 7: 48, 8: 63, 9: 83},
 'Defense': {1: 49, 2: 63, 3: 83, 4: 43, 5: 58, 6: 78, 7: 65, 8: 80, 9: 100},
 'HP': {1: 45, 2: 60, 3: 80, 4: 39, 5: 58, 6: 78, 7: 44, 8: 59, 9: 79},
 'Sp. Atk': {1: 65, 2: 80, 3: 100, 4: 60, 5: 80, 6: 109, 7: 50, 8: 65, 9: 85},
 'Sp. Def': {1: 65, 2: 80, 3: 100, 4: 50, 5: 65, 6: 85, 7: 64, 8: 80, 9: 105},
 'Speed': {1: 45, 2: 60, 3: 80, 4: 65, 5: 80, 6: 100, 7: 43, 8: 58, 9: 78}
}

pokemon = pd.DataFrame(poke_dict)

pokemon.shape

pokemon.columns

pokemon.dtypes

squirt = pokemon.tail(3)

squirt

pokemon.set_index("Name", inplace = True)

pokemon.Speed.mean()

pokemon["Balance"] = pokemon.Attack / (pokemon.HP + pokemon.Defense)

pokemon.Balance.sort_values().plot(kind = "barh", color = "blue", alpha = 0.50) #Just to make it fancy.

plt.show()

fig, ax = plt.subplots(1,2, figsize = (10,5))

fig.tight_layout(w_pad=8) 

ax[0].pie(pokemon["Sp. Atk"], labels= pokemon.index)

ax[1].bar(pokemon.index, pokemon["Sp. Atk"], color = "orange")

ax[1].tick_params(axis='x', rotation=70)

ax[1].spines["right"].set_visible(False)
ax[1].spines["top"].set_visible(False)

fig.suptitle("Special Abilities \n")

plt.show()



