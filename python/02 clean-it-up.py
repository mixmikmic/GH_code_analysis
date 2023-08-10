import pandas as pd
pd.set_option("max_columns", None)
wargame = pd.read_csv("../data/510049986/data.csv", index_col=0)

get_ipython().magic('matplotlib inline')
import seaborn as sns

wargame['AliasName'].sample(10)

wargame[wargame['AliasName'].map(lambda n: 'GR3' in n)]

wargame['AliasName'].value_counts()[0], wargame['AliasName'].value_counts()[1:].sum()

def name(srs):
    if srs['AliasName'] != 'null':
        return srs['AliasName'].title()
    else:
        return srs['ClassNameForDebug'].replace("Unit", "").replace("_", " ")

names = wargame.apply(name, axis='columns')

names

wargame['UnitName'] = names

import numpy as np
wargame = wargame.replace(to_replace='null', value=np.nan)

print(list(wargame.columns))

wargame[wargame['ClassNameForDebug'].map(lambda n: 'Beret' in n)]

leading_cols = ['UnitName', 'MotherCountry', 'ProductionPrice', 'ProductionYear',
                'MaxDamages', 'IsPrototype',
                'ManageUnitOrientation', 'CanDeploySmoke', 
                'CanWinExperience', 'ExperienceGainBySecond',
                'UnitMovingType', 'MovementType', 'SpeedBonusOnRoad', 'Maxspeed', 'FuelCapacity', 'FuelMoveDuration',
                'Transporter', 'IsTargetableAsBoat', 'TerrainsToIgnoreMask',
                'RookieDeployableAmount', 'TrainedDeployableAmount', 'HardenedDeployableAmount',
                'VeteranDeployableAmount', 'EliteDeployableAmount',
                'UnitStealthBonus', 'HitRollSizeModifier', 'HitRollECMModifier',
                'Maxspeed', 'UnitMovingType', 'FlyingAltitude', 'MinimalAltitude', 'GunMuzzleSpeed',
                'CyclicManoeuvrability', 'GFactorLimit', 'LateralSpeed', 'Mass', 'MaxInclination',
                'RotorArea', 'TorqueManoeuvrability', 'UpwardsSpeed', 'TempsDemiTour',
                'MaxAcceleration', 'MaxDeceleration', 'WeaponSabordAngle',
                'AliasName', 'ClassNameForDebug', '_ShortDatabaseName', 'NameInMenuToken',
                'Factory', 'IconeType', 'MaxPacks', 'Nationalite', 'PositionInMenu', 'TypeForAcknow',
                'VehicleSubType', 'Key', 'CoutEtoile', 'UpgradeRequire',                
                'ProductionTime',
                'Category', 'AcknowUnitType', 'TypeForAcknow', 'DescriptorId',
                'MaxHPForHUD',                
               ]
nonleading_cols = list(set(wargame.columns) - set(leading_cols))
wargame = wargame.reindex(columns=leading_cols + nonleading_cols)

wargame['IsPrototype'] = wargame['IsPrototype'].map(lambda v: True if pd.notnull(v) else False)

wargame['MotherCountry'].value_counts()

wargame[wargame['MotherCountry'] == 'RFA'].head()

mc_map = {
    'URSS': 'Soviet Union',
    'US': 'United States',
    'RDA': 'East Germany',
    'POL': 'Poland',
    'CHI': 'China',
    'NK': 'North Korea',
    'TCH': 'Czechoslovakia',
    'ISR': 'Israel',
    'UK': 'United Kingdom',
    'FR': 'France',
    'RFA': 'West Germany',
    'ROK': 'South Korea',
    'HOL': 'Netherlands',
    'ANZ': 'ANZAC',
    'CAN': 'Canada',
    'SWE': 'Sweden',
    'JAP': 'Japan',
    'DAN': 'Denmark',
    'NOR': 'Norway'    
}

wargame['MotherCountry'] = wargame['MotherCountry'].map(mc_map)

wargame['MovementType'].value_counts(dropna=False)

mv_map = {
    'TMouvementHandlerLandVehicleDescriptor': 'Land',
    'TMouvementHandlerAirplaneDescriptor': 'Airplane',
    'TMouvementHandlerHelicopterDescriptor': 'Helicopter'
}

wargame['MovementType'] = wargame['MovementType'].map(mv_map)

wargame['Transporter'] = wargame['Transporter'].map(lambda v: False if pd.isnull(v) else True)
wargame['IsTargetableAsBoat'] = wargame['IsTargetableAsBoat'].map(lambda v: False if pd.isnull(v) else True)

wargame = wargame.rename(columns={'MaxDamages': 'Health'})

wargame = wargame.rename(columns={'IsPrototype': 'Prototype'})

wargame['ManageUnitOrientation'].value_counts(dropna=False)

wargame['AutoOrientation'].value_counts(dropna=False)

wargame = wargame.T.drop_duplicates().T

wargame['HitRollSizeModifier'] = wargame['HitRollSizeModifier'].map(lambda v: v if pd.notnull(v) else 0).astype(float)
wargame['HitRollECMModifier'] = wargame['HitRollECMModifier'].map(lambda v: v if pd.notnull(v) else 0).astype(float)

wargame['TerrainsToIgnoreMask'].value_counts()

del wargame['UpgradeRequire']

del wargame['TextureForInterface']

wargame.head(5)

wargame['AcknowUnitType'].value_counts().plot(kind='bar')

