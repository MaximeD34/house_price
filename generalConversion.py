# %%
import numpy as np

# %%
def oneHotEncode(column, uniqueValues):
    oneHotEncodedColumn = np.zeros((len(column), len(uniqueValues)))
    for i in range(len(column)):
        oneHotEncodedColumn[i][list(uniqueValues).index(column[i])] = 1
    return oneHotEncodedColumn.astype(int)

# %%
def printRepartion(column, uniqueValues, message = ""):
    print(f"--- Values repartition {message} ---")
    for i in range(len(uniqueValues)):
        nb = len(column[column == uniqueValues[i]])
        print(uniqueValues[i], ":", len(column[column == uniqueValues[i]]), "(", round(nb / len(column) * 100, 1), "%)")
    print("\n")

# %%
def plotRepartition(column, uniqueValues):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.bar(uniqueValues, [len(column[column == uniqueValues[i]]) for i in range(len(uniqueValues))])
    plt.show()

# %%
#Id,MSSubClass,MSZoning,LotFrontage,LotArea,Street,Alley,LotShape,LandContour,Utilities,LotConfig,LandSlope,Neighborhood,Condition1,Condition2,BldgType,HouseStyle,OverallQual,OverallCond,YearBuilt,YearRemodAdd,RoofStyle,RoofMatl,Exterior1st,Exterior2nd,MasVnrType,MasVnrArea,ExterQual,ExterCond,Foundation,BsmtQual,BsmtCond,BsmtExposure,BsmtFinType1,BsmtFinSF1,BsmtFinType2,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,Heating,HeatingQC,CentralAir,Electrical,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,BsmtFullBath,BsmtHalfBath,FullBath,HalfBath,BedroomAbvGr,KitchenAbvGr,KitchenQual,TotRmsAbvGrd,Functional,Fireplaces,FireplaceQu,GarageType,GarageYrBlt,GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,PavedDrive,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea,PoolQC,Fence,MiscFeature,MiscVal,MoSold,YrSold,SaleType,SaleCondition,SalePrice

#==> we keep the following columns: + means we one hot encode it, = means we keep it as is
#MSSubClass +, MSZoning =, LotFrontage =, Stree +, Alley +, LotShape +,
#LandContour +, Utilities +, LotConfig +, LandSlope +, Neighborhood +,
#Condition1 +, Condition2 +, BldgType +, HouseStyle +, OverallQual =,
#OverallCond =, // YearBuilt // =, // YearRemodAdd // =, RoofStyle +, RoofMatl +, 
#Exterior1st +, Exterior2nd +, MasVnrType +, MasVnrArea =, ExterQual +,
#ExterCond +, Foundation +, BsmtQual +, BsmtCond +, BsmtExposure +,
#BsmtFinType1 +, BsmtFinSF1 =, BsmtFinType2 +, BsmtFinSF2 =, BsmtUnfSF =,
#TotalBsmtSF =, Heating +, HeatingQC +, CentralAir +, Electrical +,
#1stFlrSF =, 2ndFlrSF =, LowQualFinSF =, GrLivArea =, BsmtFullBath =,
#BsmtHalfBath =, FullBath =, HalfBath =, Bedroom =, Kitchen =, // KitchenQual //,
#TotRmsAbvGrd =, // Functional // +, Fireplaces =, // FireplaceQu // +, 
#GarageType +, // GarageYrBlt // =, // GarageFinish + //, GarageCars =, 
#GarageArea =, // GarageQual // +, // GarageCond // +, // PavedDrive // +,
#WoodDeckSF =, OpenPorchSF =, EnclosedPorch =, 3SsnPorch =, ScreenPorch =,
#PoolArea =, // PoolQC // +, // Fence // +, ?? MiscFeature ?? +, ?? MiscVal ?? =,
#// MoSold //=, // YrSold // =, SaleType +, SaleCondition +

# %%
with open("train.csv") as f:
    lines = f.readlines()

header = lines[0].split(",") #header
lines = lines[1:]

data = np.array([line.split(",") for line in lines])

#increase print size
np.set_printoptions(threshold=np.inf)

print(header)
print(data.shape)



# %%
#MSSubClass
MSSubClass = data[:, 1].copy().astype(int)
uniqueMSSubClass = np.array((20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190))
# printRepartion(MSSubClass, uniqueMSSubClass)
plotRepartition(MSSubClass, uniqueMSSubClass)

#we only keep the 4 most common values
MSSubClass = np.array([MSSubClass[i] if MSSubClass[i] in (20, 60, 50, 120) else 0 for i in range(len(MSSubClass))])
uniqueMSSubClass = np.array((0, 20, 60, 50, 120))
plotRepartition(MSSubClass, uniqueMSSubClass)

oneHotEncodedMSSubClass = oneHotEncode(MSSubClass, uniqueMSSubClass)

# %%
#MSZoning
MSZoning = data[:,2].copy()
uniqueMSZoning = np.array(["A", "C (all)", "FV", "I", "RH", "RL", "RP", "RM"])
printRepartion(MSZoning, uniqueMSZoning)

#we keep RL, RM, and others are grouped in "other"
MSZoning[MSZoning == "A"] = "other"
MSZoning[MSZoning == "C (all)"] = "other"
MSZoning[MSZoning == "FV"] = "other"
MSZoning[MSZoning == "I"] = "other"
MSZoning[MSZoning == "RH"] = "other"
MSZoning[MSZoning == "RP"] = "other"
uniqueMSZoning = np.array(["other", "RL", "RM"])

printRepartion(MSZoning, uniqueMSZoning)
plotRepartition(MSZoning, uniqueMSZoning)

oneHotEncodedMSZoning = oneHotEncode(MSZoning, uniqueMSZoning)
print(oneHotEncodedMSZoning[:10])

# %%
#LotFrontage
LotFrontage = data[:,3].copy()
LotFrontage[LotFrontage == "NA"] = np.nan
LotFrontage = LotFrontage.astype(float)
print(LotFrontage[:20])

# %%
#LotArea
LotArea = data[:,4].copy().astype(float)
print(LotArea[:20])

# %%
#TO IGNORE : Grvl is ultra rare

#Street
Street = data[:,5].copy()
uniqueStreet = np.array(["Grvl", "Pave"])

printRepartion(Street, uniqueStreet)
plotRepartition(Street, uniqueStreet)

# oneHotEncodedStreet = oneHotEncode(Street, uniqueStreet)
# print(oneHotEncodedStreet[:10]) #1 0 is rare

# %%
#Alley
Alley = data[:,6].copy()
uniqueAlley = np.array(["Grvl", "Pave", "NA"])

printRepartion(Alley, uniqueAlley)

#We keep only NA and other
Alley[Alley == "Grvl"] = "other"
Alley[Alley == "Pave"] = "other"
uniqueAlley = np.array(["NA", "other"])

printRepartion(Alley, uniqueAlley, "after cleaning")

oneHotEncodedAlley = oneHotEncode(Alley, uniqueAlley)
print(oneHotEncodedAlley[:10]) #almost all NA

# %%
#LotShape
LotShape = data[:,7].copy()
lotshape_mapping = {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3}

printRepartion(LotShape, list(lotshape_mapping.keys()))

#We only keep Reg and others are grouped in "other"
LotShape[LotShape == "IR1"] = "other"
LotShape[LotShape == "IR2"] = "other"
LotShape[LotShape == "IR3"] = "other"
lotshape_mapping = {"Reg": 0, "other": 1}

printRepartion(LotShape, list(lotshape_mapping.keys()), "after cleaning")

LotShape = np.array([lotshape_mapping[shape] for shape in LotShape])
print(data[:20, 7])
print(LotShape[:20])

# %%
#LandContour
LandContour = data[:,8].copy()
landcontour_mapping = {"Lvl": 0, "Bnk": 1, "HLS": 2, "Low": 3}

printRepartion(LandContour, list(landcontour_mapping.keys()))

#We only keep Lvl and others are grouped in "other"
LandContour[LandContour != "Lvl"] = "other"
landcontour_mapping = {"Lvl": 0, "other": 1}
printRepartion(LandContour, list(landcontour_mapping.keys()), "after cleaning")

LandContour = np.array([landcontour_mapping[contour] for contour in LandContour])
print(data[:10, 8])
print(LandContour[:100])

# %%
#TO IGNORE : AllPub is ultra majoritary

#Utilities
Utilities = data[:,9].copy()
utilities_mapping = {"AllPub": 0, "NoSewr": 1, "NoSeWa": 2, "ELO": 3}

printRepartion(Utilities, list(utilities_mapping.keys()))

# Utilities = np.array([utilities_mapping[utility] for utility in Utilities])
# print(data[:10, 9])
# print(Utilities[:10])

# %%
#TO DECIDE : we keep it or not ?

#LotConfig
LotConfig = data[:,10].copy()
uniqueLotConfig = np.array(["Inside", "Corner", "CulDSac", "FR2", "FR3"])

printRepartion(LotConfig, uniqueLotConfig)
plotRepartition(LotConfig, uniqueLotConfig)

#we only keep the 4 most common values$
LotConfig[LotConfig == "FR2"] = "other"
LotConfig[LotConfig == "FR3"] = "other"
uniqueLotConfig = np.array(["Inside", "Corner", "CulDSac", "other"])

printRepartion(LotConfig, uniqueLotConfig)
plotRepartition(LotConfig, uniqueLotConfig)

oneHotEncodedLotConfig = oneHotEncode(LotConfig, uniqueLotConfig)

print(oneHotEncodedLotConfig[:10])


# %%
#TO IGNORE : Sev is ultra rare and Mod is rare

#LandSlope
LandSlope = data[:,11].copy()
landslope_mapping = {"Gtl": 0, "Mod": 1, "Sev": 2}

printRepartion(LandSlope, list(landslope_mapping.keys()))

#We only keep Gtl and others are grouped in "other"
LandSlope[LandSlope != "Gtl"] = "other"
landslope_mapping = {"Gtl": 0, "other": 1}

printRepartion(LandSlope, list(landslope_mapping.keys()), "after cleaning")

# LandSlope = np.array([landslope_mapping[slope] for slope in LandSlope])
# print(data[:100, 11])
# print(LandSlope[:100])

# %%
#TO CHECK : we keep it or not ?

#Neighborhood
Neighborhood = data[:,12].copy()
uniqueNeighborhood = np.array(["Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", 
                               "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR", 
                               "MeadowV", "Mitchel", "NAmes", "NoRidge", "NPkVill", 
                               "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer", 
                               "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker"])

# printRepartion(Neighborhood, uniqueNeighborhood)
plotRepartition(Neighborhood, uniqueNeighborhood)

#we only keep the categories with more than 90 houses
for value in uniqueNeighborhood:
    if len(Neighborhood[Neighborhood == value]) < 80:
        Neighborhood[Neighborhood == value] = "other"
        uniqueNeighborhood = uniqueNeighborhood[uniqueNeighborhood != value]

uniqueNeighborhood = np.append(uniqueNeighborhood, "other")

# printRepartion(Neighborhood, uniqueNeighborhood)
plotRepartition(Neighborhood, uniqueNeighborhood)
printRepartion(Neighborhood, uniqueNeighborhood, "after cleaning")

oneHotEncodedNeighborhood = oneHotEncode(Neighborhood, uniqueNeighborhood)
print(oneHotEncodedNeighborhood[:10])


# %%
#Condition1
Condition1 = data[:,13].copy()
uniqueCondition1 = np.array(["Artery", "Feedr", "Norm", "RRNn",
                             "RRAn", "PosN", "PosA", "RRNe", "RRAe"])

printRepartion(Condition1, uniqueCondition1, "condition 1 before cleaning")

#we only keep Norm and others are grouped in "other"
Condition1[Condition1 != "Norm"] = "other"
uniqueCondition1 = np.array(["Norm", "other"])

printRepartion(Condition1, uniqueCondition1, "condition 1 after cleaning")

oneHotEncodedCondition1 = oneHotEncode(Condition1, uniqueCondition1)

#TO REMOVE : Norm is ultra majoritary
#Condition2
Condition2 = data[:,14]
uniqueCondition2 = np.array(["Artery", "Feedr", "Norm", "RRNn",
                             "RRAn", "PosN", "PosA", "RRNe", "RRAe"])
# printRepartion(Condition2, uniqueCondition2, "condition 2 before cleaning")
# oneHotEncodedCondition2 = oneHotEncode(Condition2, uniqueCondition2)

# #union of condition1 and condition2
# oneHotEncodedCondition = np.logical_or(oneHotEncodedCondition1, oneHotEncodedCondition2).astype(int)

print(oneHotEncodedCondition1[:10])



# %%
#BldgType
BldgType = data[:,15].copy()
uniqueBldgType = np.array(["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"])

printRepartion(BldgType, uniqueBldgType)
plotRepartition(BldgType, uniqueBldgType)

#we only keep 1Fam and others are grouped in "other"
BldgType[BldgType != "1Fam"] = "other"
uniqueBldgType = np.array(["1Fam", "other"])

printRepartion(BldgType, uniqueBldgType, "after cleaning")
plotRepartition(BldgType, uniqueBldgType)

oneHotEncodedBldgType = oneHotEncode(BldgType, uniqueBldgType)
print(oneHotEncodedBldgType[:10])


# %%
#HouseStyle
HouseStyle = data[:,16].copy()
uniqueHouseStyle = np.array(["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"])

printRepartion(HouseStyle, uniqueHouseStyle)
plotRepartition(HouseStyle, uniqueHouseStyle)

#we only keep the first 3 categories
HouseStyle[~np.isin(HouseStyle,  ["1Story", "1.5Fin", "2Story"])] = "other"
uniqueHouseStyle = np.array(["1Story", "1.5Fin", "2Story", "other"])

printRepartion(HouseStyle, uniqueHouseStyle, "after cleaning")
plotRepartition(HouseStyle, uniqueHouseStyle)
oneHotEncodedHouseStyle = oneHotEncode(HouseStyle, uniqueHouseStyle)
print(oneHotEncodedHouseStyle[:10])

# %%
#OverallQual
OverallQual = data[:,17].copy().astype(int)
print(OverallQual[:10])

#OverallCond
OverallCond = data[:,18].copy().astype(int)
print(OverallCond[:10])


# %%
#Get sold year
YearSold = data[:,77].copy().astype(int)
print(YearSold[:10])

# %%
#YearBuilt
YearBuilt = data[:,19].copy().astype(int)
print(YearBuilt[:10])

delta_YearSold_YearBuilt = YearSold - YearBuilt
print(delta_YearSold_YearBuilt[:10])

# %%
#YearRemodAdd
YearRemodAdd = data[:,20].copy().astype(int)

delta_YearSold_YearRemodAdd = YearSold - YearRemodAdd
print(delta_YearSold_YearRemodAdd[:10])

# %%
#RoofStyle
RoofStyle = data[:,21].copy()
uniqueRoofStyle = np.array(["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"])

printRepartion(RoofStyle, uniqueRoofStyle)
plotRepartition(RoofStyle, uniqueRoofStyle)

#we only keep Gable and others are grouped in "other"
RoofStyle[RoofStyle != "Gable"] = "other"
uniqueRoofStyle = np.array(["Gable", "other"])

printRepartion(RoofStyle, uniqueRoofStyle, "after cleaning")
plotRepartition(RoofStyle, uniqueRoofStyle)

oneHotEncodedRoofStyle = oneHotEncode(RoofStyle, uniqueRoofStyle)
print(oneHotEncodedRoofStyle[:10])

# %%
#TO IGNORE : CompShg is ultra majoritary

#RoofMatl
RoofMatl = data[:,22].copy()
uniqueRoofMatl = np.array(["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", 
                           "WdShake", "WdShngl"])

printRepartion(RoofMatl, uniqueRoofMatl)
plotRepartition(RoofMatl, uniqueRoofMatl)

# oneHotEncodedRoofMatl = oneHotEncode(RoofMatl, uniqueRoofMatl)
# print(oneHotEncodedRoofMatl[:10])

# %%
#Exterior1st
Exterior1st = data[:,23].copy()
Exterior1st[Exterior1st == "Wd Shng"] = "WdShing"
Exterior1st[Exterior1st == "Brk Cmn"] = "BrkComm"
Exterior1st[Exterior1st == "CmentBd"] = "CemntBd"

uniqueExterior = np.array(["AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", 
                              "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", 
                              "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", 
                              "Wd Sdng", "WdShing"])

# printRepartion(Exterior1st, uniqueExterior)
# plotRepartition(Exterior1st, uniqueExterior)

#we only keep the values with more than 100 houses
for value in uniqueExterior:
    if len(Exterior1st[Exterior1st == value]) < 100:
        Exterior1st[Exterior1st == value] = "other"
        uniqueExterior = uniqueExterior[uniqueExterior != value]
uniqueExterior = np.append(uniqueExterior, "other")

printRepartion(Exterior1st, uniqueExterior, "after cleaning")
plotRepartition(Exterior1st, uniqueExterior)

oneHotEncodedExterior1st = oneHotEncode(Exterior1st, uniqueExterior)

#Exterior2nd
Exterior2nd = data[:,24].copy()
Exterior2nd[Exterior2nd == "Wd Shng"] = "WdShing"
Exterior2nd[Exterior2nd == "Brk Cmn"] = "BrkComm"
Exterior2nd[Exterior2nd == "CmentBd"] = "CemntBd"

# printRepartion(Exterior2nd, uniqueExterior)
# plotRepartition(Exterior2nd, uniqueExterior)

Exterior2nd[~np.isin(Exterior2nd,  uniqueExterior)] = "other"

printRepartion(Exterior2nd, uniqueExterior, "after cleaning")
plotRepartition(Exterior2nd, uniqueExterior)

oneHotEncodedExterior2nd = oneHotEncode(Exterior2nd, uniqueExterior)

#union of exterior1st and exterior2nd
oneHotEncodedExterior = np.logical_or(oneHotEncodedExterior1st, oneHotEncodedExterior2nd).astype(int)
print(oneHotEncodedExterior[:10])

# %%
#MasVnrType
MasVnrType = data[:,25].copy()
MasVnrType[MasVnrType == "None"] = "NA"
uniqueMasVnrType = np.array(["BrkCmn", "BrkFace", "CBlock", "NA", "Stone"])

printRepartion(MasVnrType, uniqueMasVnrType)
plotRepartition(MasVnrType, uniqueMasVnrType)

#we only keep NA and BrkFace and others are grouped in "other"
MasVnrType[~np.isin(MasVnrType,  ["NA", "BrkFace"])] = "other"
uniqueMasVnrType = np.array(["NA", "BrkFace", "other"])

printRepartion(MasVnrType, uniqueMasVnrType, "after cleaning")
plotRepartition(MasVnrType, uniqueMasVnrType)

oneHotEncodedMasVnrType = oneHotEncode(MasVnrType, uniqueMasVnrType)
print(oneHotEncodedMasVnrType[:10])

# %%
#MasVnrArea
MasVnrArea = data[:,26].copy()
MasVnrArea[MasVnrArea == "NA"] = np.nan
MasVnrArea = MasVnrArea.astype(float)
print(MasVnrArea[:100])

# %%
#ExterQual
ExterQual = data[:,27].copy()
exterqual_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}
ExterQual = np.array([exterqual_mapping[qual] for qual in ExterQual])

printRepartion(ExterQual, list(exterqual_mapping.values()))
plotRepartition(ExterQual, list(exterqual_mapping.values()))

#we only keep 1-2 , 0 is merged with 1 and the others are merged in 2
ExterQual[ExterQual == 0] = 1
ExterQual[ExterQual > 2] = 2
exterqual_mapping = {"Ex, Gd": 1, "TA, Fa, Po": 2}

printRepartion(ExterQual, list(exterqual_mapping.values()), "after cleaning")
plotRepartition(ExterQual, list(exterqual_mapping.values()))

print(data[:10, 27])
print(ExterQual[:10])

# %%
#ExterCond
ExterCond = data[:,28].copy()
extercond_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}
ExterCond = np.array([extercond_mapping[cond] for cond in ExterCond])

printRepartion(ExterCond, list(extercond_mapping.values()))
plotRepartition(ExterCond, list(extercond_mapping.values()))

#0 <= [0, 1] and 1 <= [2, 3, 4] 
ExterCond[ExterCond <= 1] = 0
ExterCond[ExterCond > 1] = 1
extercond_mapping = {"Ex, Gd": 0, "TA, Fa, Po": 1}

printRepartion(ExterCond, list(extercond_mapping.values()), "after cleaning")
plotRepartition(ExterCond, list(extercond_mapping.values()))

print(data[:10, 28])
print(ExterCond[:10])

# %%
#Foundation
Foundation = data[:,29].copy()
uniqueFoundation = np.array(["BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"])

printRepartion(Foundation, uniqueFoundation)
plotRepartition(Foundation, uniqueFoundation)

#we only keep PConc and CBlock and others are grouped in "other"
Foundation[~np.isin(Foundation,  ["PConc", "CBlock"])] = "other"
uniqueFoundation = np.array(["PConc", "CBlock", "other"])

printRepartion(Foundation, uniqueFoundation, "after cleaning")
plotRepartition(Foundation, uniqueFoundation)

oneHotEncodedFoundation = oneHotEncode(Foundation, uniqueFoundation)
print(oneHotEncodedFoundation[:10])


# %%
#BsmtQual
BsmtQual = data[:,30].copy()
BsmtQual_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
BsmtQual = np.array([BsmtQual_mapping[qual] for qual in BsmtQual])

printRepartion(BsmtQual, list(BsmtQual_mapping.values()))
plotRepartition(BsmtQual, list(BsmtQual_mapping.values()))

#0 <= [0, 1] and 1 <= [2, 3, 4]
BsmtQual[BsmtQual <= 1] = 0
BsmtQual[BsmtQual > 1] = 1

BsmtQual_mapping = {"Ex, Gd": 0, "TA, Fa, Po, NA": 1}

printRepartion(BsmtQual, list(BsmtQual_mapping.values()), "after cleaning")
plotRepartition(BsmtQual, list(BsmtQual_mapping.values()))

print(data[:10, 30])
print(BsmtQual[:10])

# %%
#TO CHECK : we keep it or not ?

#BsmtCond
BsmtCond = data[:,31].copy()
BsmtCond_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
BsmtCond = np.array([BsmtCond_mapping[cond] for cond in BsmtCond])

printRepartion(BsmtCond, list(BsmtCond_mapping.values()))
plotRepartition(BsmtCond, list(BsmtCond_mapping.values()))

#0 <= [0, 1] and 1 <= [2, 3, 4]
BsmtCond[BsmtCond <= 1] = 0
BsmtCond[BsmtCond > 1] = 1

BsmtCond_mapping = {"Ex, Gd": 0, "TA, Fa, Po, NA": 1}

printRepartion(BsmtCond, list(BsmtCond_mapping.values()), "after cleaning")
plotRepartition(BsmtCond, list(BsmtCond_mapping.values()))

print(data[:10, 31])
print(BsmtCond[:10])

# %%
#BsmtExposure
BsmtExposure = data[:,32].copy()
BsmtExposure_mapping = {"Gd": 0, "Av": 1, "Mn": 2, "No": 3, "NA": 4}
BsmtExposure = np.array([BsmtExposure_mapping[exposure] for exposure in BsmtExposure])

printRepartion(BsmtExposure, list(BsmtExposure_mapping.values()))
plotRepartition(BsmtExposure, list(BsmtExposure_mapping.values()))

#TO CHECK : does it make sense to have a gradation with NA in it ?

#3 <= [3,4]
BsmtExposure[BsmtExposure == 4] = 3
BsmtExposure_mapping = {"Gd": 0, "Av": 1, "Mn": 2, "No, NA": 3}

printRepartion(BsmtExposure, list(BsmtExposure_mapping.values()), "after cleaning")
plotRepartition(BsmtExposure, list(BsmtExposure_mapping.values()))

print(data[:10, 32])
print(BsmtExposure[:10])

# %%
#BsmtFinType1
BsmtFinType1 = data[:,33].copy()
BsmtFinType1_mapping = {"GLQ": 0, "ALQ": 1, "BLQ": 2, "Rec": 3
                        , "LwQ": 4, "Unf": 5, "NA": 6}
BsmtFinType1 = np.array([BsmtFinType1_mapping[finType] for finType in BsmtFinType1])

printRepartion(BsmtFinType1, list(BsmtFinType1_mapping.values()))
plotRepartition(BsmtFinType1, list(BsmtFinType1_mapping.values()))

#TO CHECK : should we group more ?

#5 <= [5, 6]
BsmtFinType1[BsmtFinType1 == 6] = 5
BsmtFinType1_mapping = {"GLQ": 0, "ALQ": 1, "BLQ": 2, "Rec": 3
                        , "LwQ": 4, "Unf, NA": 5}

printRepartion(BsmtFinType1, list(BsmtFinType1_mapping.values()), "after cleaning")
plotRepartition(BsmtFinType1, list(BsmtFinType1_mapping.values()))


print(data[:10, 33])
print(BsmtFinType1[:10])

# %%
#BsmtFinSF1
BsmtFinSF1 = data[:,34].copy().astype(float)
print(BsmtFinSF1[:10])

# %%
#BsmtFinType2
BsmtFinType2 = data[:,35].copy()
BsmtFinType2_mapping = {"GLQ": 0, "ALQ": 1, "BLQ": 2, "Rec": 3, "LwQ": 4, "Unf": 5, "NA": 6}
BsmtFinType2 = np.array([BsmtFinType2_mapping[finType] for finType in BsmtFinType2])
printRepartion(BsmtFinType2, list(BsmtFinType2_mapping.values()))
plotRepartition(BsmtFinType2, list(BsmtFinType2_mapping.values()))

#0 <= [0:4] and 1 <= [5, 6]
BsmtFinType2[BsmtFinType2 <= 4] = 0
BsmtFinType2[BsmtFinType2 > 4] = 1
BsmtFinType2_mapping = {"GLQ, ALQ, BLQ, Rec, LwQ": 0, "Unf, NA": 1}

printRepartion(BsmtFinType2, list(BsmtFinType2_mapping.values()), "after cleaning")
plotRepartition(BsmtFinType2, list(BsmtFinType2_mapping.values()))


# %%
#BsmtFinSF2
BsmtFinSF2 = data[:,36].copy().astype(float)
print(BsmtFinSF2[:10])

# %%
#BsmtUnfSF
BsmtUnfSF = data[:,37].copy().astype(float)
print(BsmtUnfSF[:10])

# %%
#TotalBsmtSF
TotalBsmtSF = data[:,38].copy().astype(float)
print(TotalBsmtSF[:10])

# %%
#TO IGNORE : GasA is ultra majoritary

#Heating
Heating = data[:,39]
uniqueHeating = np.array(["Floor", "GasA", "GasW", "Grav", "OthW", "Wall"])

printRepartion(Heating, uniqueHeating)
plotRepartition(Heating, uniqueHeating)

# oneHotEncodedHeating = oneHotEncode(Heating, uniqueHeating)
# print(oneHotEncodedHeating[:10])
# print(Heating[Heating == "Wall"])


# %%
#HeatingQC
HeatingQC = data[:,40].copy()
heatingqc_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4} #group Po with Fa because of low number of Po and Fa
HeatingQC = np.array([heatingqc_mapping[qual] for qual in HeatingQC])

printRepartion(HeatingQC, list(heatingqc_mapping.values()))
plotRepartition(HeatingQC, list(heatingqc_mapping.values()))

#2 <= [2, 3, 4]
HeatingQC[HeatingQC > 1] = 2
heatingqc_mapping = {"Ex": 0, "Gd": 1, "TA, Fa, Po": 2}

printRepartion(HeatingQC, list(heatingqc_mapping.values()), "after cleaning")
plotRepartition(HeatingQC, list(heatingqc_mapping.values()))

print(data[:10, 40])
print(HeatingQC[:10])

# %%
#CentralAir
CentralAir = data[:,41].copy()
uniqueCentralAir = np.array(["N", "Y"])

printRepartion(CentralAir, uniqueCentralAir)
plotRepartition(CentralAir, uniqueCentralAir)

oneHotEncodedCentralAir = oneHotEncode(CentralAir, uniqueCentralAir)
print(oneHotEncodedCentralAir[:10])

# %%
#Electrical
Electrical = data[:,42].copy()
uniqueElectrical = np.array(["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"])

printRepartion(Electrical, uniqueElectrical)
plotRepartition(Electrical, uniqueElectrical)

#we only keep SBrkr and others are grouped in Fuse
Electrical[Electrical != "SBrkr"] = "Fuse"
uniqueElectrical = np.array(["SBrkr", "Fuse"])

printRepartion(Electrical, uniqueElectrical, "after cleaning")
plotRepartition(Electrical, uniqueElectrical)





# %%
#1stFlrSF
FirstFlrSF = data[:,43].copy().astype(float)
print(FirstFlrSF[:10])

# %%
#2ndFlrSF
SecondFlrSF = data[:,44].copy().astype(float)
print(SecondFlrSF[:10])

# %%
#LowQualFinSF
LowQualFinSF = data[:,45].copy().astype(float)
print(LowQualFinSF[:10])

# %%
#GrLivArea
GrLivArea = data[:,46].copy().astype(float)
print(GrLivArea[:10])

# %%
#BsmtFullBath
BsmtFullBath = data[:,47].copy().astype(float)
print(BsmtFullBath[:100])

# %%
#BsmtHalfBath
BsmtHalfBath = data[:,48].copy().astype(float)
print(BsmtHalfBath[:100])

# %%
#FullBath
FullBath = data[:,49].copy().astype(float)
print(FullBath[:100])

# %%
#HalfBath
HalfBath = data[:,50].copy().astype(float)
print(HalfBath[:100])

# %%
#Bedromm 
Bedroom = data[:,51].copy().astype(float)
print(Bedroom[:100])

# %%
#Kitchen
Kitchen = data[:,52].copy().astype(float)
print(Kitchen[:100])

# %%
#KitchenQual
KitchenQual = data[:,53].copy()
kitchenqual_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}
KitchenQual = np.array([kitchenqual_mapping[qual] for qual in KitchenQual])

printRepartion(KitchenQual, list(kitchenqual_mapping.values()))
plotRepartition(KitchenQual, list(kitchenqual_mapping.values()))

#0 <= [0, 1] and 1 <= [2, 3, 4]
KitchenQual[KitchenQual <= 1] = 0
KitchenQual[KitchenQual > 1] = 1
kitchenqual_mapping = {"Ex, Gd": 0, "TA, Fa, Po": 1}

printRepartion(KitchenQual, list(kitchenqual_mapping.values()), "after cleaning")
plotRepartition(KitchenQual, list(kitchenqual_mapping.values()))

# %%
#TotRmsAbvGrd
TotRmsAbvGrd = data[:,54].copy().astype(float)
print(TotRmsAbvGrd[:100])

# %%
#Functional
Functional = data[:,55].copy()
functional_mapping = {"Typ": 0, "Min1": 1, "Min2": 2, "Mod": 3, 
                      "Maj1": 4, "Maj2": 5, "Sev": 6, "Sal": 7}

Functional = np.array([functional_mapping[func] for func in Functional])

printRepartion(Functional, list(functional_mapping.values()))
plotRepartition(Functional, list(functional_mapping.values()))

#we only keep Typ and others are grouped in "other"
Functional[Functional != 0] = 1
functional_mapping = {"Typ": 0, "other": 1}

printRepartion(Functional, list(functional_mapping.values()), "after cleaning")
plotRepartition(Functional, list(functional_mapping.values()))



# %%
#Fireplaces
Fireplaces = data[:,56].copy().astype(float)
print(Fireplaces[:100])

# %%
#FireplaceQu
FireplaceQu = data[:,57].copy()
fireplacequ_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
FireplaceQu = np.array([fireplacequ_mapping[qual] for qual in FireplaceQu])

printRepartion(FireplaceQu, list(fireplacequ_mapping.values()))
plotRepartition(FireplaceQu, list(fireplacequ_mapping.values()))

#0 <= [0, 1] and 1 <= [2] and 2 <= [3, 4, 5]
FireplaceQu[FireplaceQu <= 1] = 0
FireplaceQu[FireplaceQu == 2] = 1
FireplaceQu[FireplaceQu > 2] = 2
fireplacequ_mapping = {"Ex, Gd": 0, "TA": 1, "Fa, Po, NA": 2}


printRepartion(FireplaceQu, list(fireplacequ_mapping.values()), "after cleaning")
plotRepartition(FireplaceQu, list(fireplacequ_mapping.values()))

oneHotEncodedFireplaceQu = oneHotEncode(FireplaceQu, list(fireplacequ_mapping.values()))
print(oneHotEncodedFireplaceQu[:10])



# %%
#GarageType
GarageType = data[:,58].copy()
uniqueGarageType = np.array(["2Types", "Attchd", "Basment", "BuiltIn", 
                             "CarPort", "Detchd"])

printRepartion(GarageType, uniqueGarageType)
plotRepartition(GarageType, uniqueGarageType)

#we only keep Attchd, Detchd, BuiltIn and others are grouped in "other"
GarageType[~np.isin(GarageType,  ["Attchd", "Detchd", "BuiltIn"])] = "other"
uniqueGarageType = np.array(["Attchd", "Detchd", "BuiltIn", "other"])

printRepartion(GarageType, uniqueGarageType, "after cleaning")
plotRepartition(GarageType, uniqueGarageType)

oneHotEncodedGarageType = oneHotEncode(GarageType, uniqueGarageType)
print(oneHotEncodedGarageType[:10])

# %%
#GarageYrBlt
GarageYrBlt = data[:,59].copy()
GarageYrBlt[GarageYrBlt == "NA"] = np.nan
GarageYrBlt = GarageYrBlt.astype(float)

delta_YearSold_GarageYrBlt = YearSold - GarageYrBlt

print(GarageYrBlt[:100])
print(delta_YearSold_GarageYrBlt[:100])

# %%
#GarageFinish
GarageFinish = data[:,60].copy()
garagefinish_mapping = {"Fin": 0, "RFn": 1, "Unf": 2, "NA": 3}
GarageFinish = np.array([garagefinish_mapping[finish] for finish in GarageFinish])

printRepartion(GarageFinish, list(garagefinish_mapping.values()))
plotRepartition(GarageFinish, list(garagefinish_mapping.values()))

# %%
#GarageCars
GarageCars = data[:,61].copy().astype(float)
print(GarageCars[:100])

# %%
#GarageArea
GarageArea = data[:,62].copy().astype(float)
print(GarageArea[:100])

# %%
#GarageQual
GarageQual = data[:,63].copy()
garagequal_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
GarageQual = np.array([garagequal_mapping[qual] for qual in GarageQual])

printRepartion(GarageQual, list(garagequal_mapping.values()))
plotRepartition(GarageQual, list(garagequal_mapping.values()))

#0 <= [0, 1, 2] and 1 <= [3, 4, 5]
GarageQual[GarageQual <= 2] = 0
GarageQual[GarageQual > 2] = 1
garagequal_mapping = {"Ex, Gd, TA": 0, "Fa, Po, NA": 1}

printRepartion(GarageQual, list(garagequal_mapping.values()), "after cleaning")
plotRepartition(GarageQual, list(garagequal_mapping.values()))

# %%
#GarageCond
GarageCond = data[:,64].copy()
garagecond_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}
GarageCond = np.array([garagecond_mapping[cond] for cond in GarageCond])

printRepartion(GarageCond, list(garagecond_mapping.values()))
plotRepartition(GarageCond, list(garagecond_mapping.values()))

#0 <= [0, 1, 2] and 1 <= [3, 4, 5]
GarageCond[GarageCond <= 2] = 0
GarageCond[GarageCond > 2] = 1
garagecond_mapping = {"Ex, Gd, TA": 0, "Fa, Po, NA": 1}

printRepartion(GarageCond, list(garagecond_mapping.values()), "after cleaning")
plotRepartition(GarageCond, list(garagecond_mapping.values()))

# %%
#PavedDrive
PavedDrive = data[:,65].copy()
paveddrive_mapping = {"Y": 0, "P": 1, "N": 2}
PavedDrive = np.array([paveddrive_mapping[drive] for drive in PavedDrive])

printRepartion(PavedDrive, list(paveddrive_mapping.values()))
plotRepartition(PavedDrive, list(paveddrive_mapping.values()))

#1 <= [1, 2]
PavedDrive[PavedDrive > 0] = 1
paveddrive_mapping = {"Y": 0, "P, N": 1}

printRepartion(PavedDrive, list(paveddrive_mapping.values()), "after cleaning")
plotRepartition(PavedDrive, list(paveddrive_mapping.values()))


# %%
#WoodDeckSF
WoodDeckSF = data[:,66].copy().astype(float)
print(WoodDeckSF[:100])

# %%
#OpenPorchSF
OpenPorchSF = data[:,67].copy().astype(float)
print(OpenPorchSF[:100])

# %%
#EnclosedPorch
EnclosedPorch = data[:,68].copy().astype(float)
print(EnclosedPorch[:100])

# %%
#3SsnPorch
ThreeSsnPorch = data[:,69].copy().astype(float)
print(ThreeSsnPorch[:100])

# %%
#ScreenPorch
ScreenPorch = data[:,70].copy().astype(float)
print(ScreenPorch[:100])

# %%
#TO CHECK : most of the values are 0, we keep it or not ?

#PoolArea
PoolArea = data[:,71].copy().astype(float)
print(PoolArea[:100])

print(PoolArea[PoolArea > 0])

# %%
#TO IGNORE : NA is ultra majoritary

#PoolQC
PoolQC = data[:,72].copy()
poolqc_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "NA": 4}
PoolQC = np.array([poolqc_mapping[qual] for qual in PoolQC])

printRepartion(PoolQC, list(poolqc_mapping.values()))
plotRepartition(PoolQC, list(poolqc_mapping.values()))

PoolQC = np.nan  #to ignore

# %%
#TO CHECK : weird repartition, we keep it or not ?

#Fence
Fence = data[:,73].copy()
fence_mapping = {"GdPrv": 0, "MnPrv": 1, "GdWo": 2, "MnWw": 3, "NA": 4}
Fence = np.array([fence_mapping[qual] for qual in Fence])

printRepartion(Fence, list(fence_mapping.values()))
plotRepartition(Fence, list(fence_mapping.values()))

# %%
#TO IGNORE : NA is ultra majoritary

#MiscFeature
MiscFeature = data[:,74].copy()
uniqueMiscFeature = np.array(["Elev", "Gar2", "Othr", "Shed", "TenC", "NA"])

printRepartion(MiscFeature, uniqueMiscFeature)
plotRepartition(MiscFeature, uniqueMiscFeature)

MiscFeature = np.nan #to ignore

# %%
#TO CHECK : most of the values are 0, we keep it or not ?
#TO CHECK : I removed the MiscFeature, can this work alone ?

#MiscVal
MiscVal = data[:,75].copy().astype(float)
print(MiscVal[:100])


# %%
#TO IGNORE : Seems irrelevant

#MoSold
MoSold = data[:,76].copy().astype(float)
print(MoSold[:100])

MoSold = np.nan #to ignore


# %%
#SaleType
SaleType = data[:,78].copy()
uniqueSaleType = np.array(["WD", "CWD", "VWD", "New", "COD", "Con", 
                           "ConLw", "ConLI", "ConLD", "Oth"])

printRepartion(SaleType, uniqueSaleType)
plotRepartition(SaleType, uniqueSaleType)

#we only keep WD, New, COD and others are grouped in "other"
SaleType[~np.isin(SaleType,  ["WD", "New"])] = "other"
uniqueSaleType = np.array(["WD", "New", "other"])

printRepartion(SaleType, uniqueSaleType, "after cleaning")
plotRepartition(SaleType, uniqueSaleType)

oneHotEncodedSaleType = oneHotEncode(SaleType, uniqueSaleType)
print(oneHotEncodedSaleType[:10])


# %%
#TO CHECK : is other too small ?

#SaleCondition
SaleCondition = data[:,79].copy()
uniqueSaleCondition = np.array(["Normal", "Abnorml", "AdjLand", "Alloca", 
                                "Family", "Partial"])

printRepartion(SaleCondition, uniqueSaleCondition)
plotRepartition(SaleCondition, uniqueSaleCondition)

#we only keep Normal, Abnorml, Partial and others are grouped in "other"
SaleCondition[~np.isin(SaleCondition,  ["Normal", "Abnorml", "Partial"])] = "other"
uniqueSaleCondition = np.array(["Normal", "Abnorml", "Partial", "other"])

printRepartion(SaleCondition, uniqueSaleCondition, "after cleaning")
plotRepartition(SaleCondition, uniqueSaleCondition)

oneHotEncodedSaleCondition = oneHotEncode(SaleCondition, uniqueSaleCondition)
print(oneHotEncodedSaleCondition[:10])


# %%
#MERGE ALL FEATURES

finalMatrix = np.concatenate((oneHotEncodedMSSubClass, oneHotEncodedMSZoning, LotFrontage.reshape(-1,1),
                              LotArea.reshape(-1,1), oneHotEncodedAlley,
                              LotShape.reshape(-1,1), LandContour.reshape(-1,1), oneHotEncodedLotConfig,
                               oneHotEncodedNeighborhood , oneHotEncodedCondition1,
                                 oneHotEncodedBldgType, oneHotEncodedHouseStyle, OverallQual.reshape(-1,1),
                                    OverallCond.reshape(-1,1), delta_YearSold_YearBuilt.reshape(-1,1),
                                        delta_YearSold_YearRemodAdd.reshape(-1,1), oneHotEncodedRoofStyle,
                                            oneHotEncodedExterior, oneHotEncodedMasVnrType, MasVnrArea.reshape(-1,1),
                                            ExterQual.reshape(-1,1), ExterCond.reshape(-1,1), oneHotEncodedFoundation,
                                                BsmtQual.reshape(-1,1), BsmtCond.reshape(-1,1), BsmtExposure.reshape(-1,1),
                                                BsmtFinType1.reshape(-1,1), BsmtFinSF1.reshape(-1,1), BsmtFinType2.reshape(-1,1),
                                                    BsmtFinSF2.reshape(-1,1), BsmtUnfSF.reshape(-1,1), TotalBsmtSF.reshape(-1,1),
                                                    HeatingQC.reshape(-1,1), oneHotEncodedCentralAir,
                                                        FirstFlrSF.reshape(-1,1), SecondFlrSF.reshape(-1,1),
                                                        LowQualFinSF.reshape(-1,1), GrLivArea.reshape(-1,1), BsmtFullBath.reshape(-1,1),
                                                            BsmtHalfBath.reshape(-1,1), FullBath.reshape(-1,1), HalfBath.reshape(-1,1),
                                                            Bedroom.reshape(-1,1), Kitchen.reshape(-1,1), KitchenQual.reshape(-1,1),
                                                                TotRmsAbvGrd.reshape(-1,1), Fireplaces.reshape(-1,1),
                                                                oneHotEncodedFireplaceQu, oneHotEncodedGarageType, GarageYrBlt.reshape(-1,1),
                                                                    GarageFinish.reshape(-1,1), GarageCars.reshape(-1,1), GarageArea.reshape(-1,1),
                                                                    GarageQual.reshape(-1,1), GarageCond.reshape(-1,1),                                                                         WoodDeckSF.reshape(-1,1), OpenPorchSF.reshape(-1,1), EnclosedPorch.reshape(-1,1),
                                                                        ThreeSsnPorch.reshape(-1,1), ScreenPorch.reshape(-1,1), PoolArea.reshape(-1,1),
                                                                         oneHotEncodedSaleType,  Fence.reshape(-1,1), MiscVal.reshape(-1,1),
                                                                         oneHotEncodedSaleCondition), axis=1)


print(finalMatrix.shape)

finalMatrix[0]


