'''https://arxiv.org/html/2308.06198v3#S3'''

'''https://github.com/facebookresearch/DIG-In/tree/main'''

#prompts: 


#1. {object}

#2. {object} in {country}

#3. {object} in {region}


regions = ['Africa', 'Asia', 'Europe', 'North America', 'South America', ]


geo_bias_top_20 = [
    "refrigerator",
    "alcoholic drinks",
    "car",
    "bike",
    "cooking pots",
    "freezer",
    "toilet",
    "bathroom/toilet",
    "washing detergent",
    "source of heat",
    "source of cool",
    "toothbrush",
    "cooking utensils",
    "shower",
    "menstruation pads / tampax",
    "dishwasher",
    "mosquito protection",
    "fruit trees",
    "phone",
    "trash/waste"
]


best_prompts = []
for object in geo_bias_top_20:
    best_prompts += [f'{object.replace("/","-")} in {region}' for region in regions]



with open('top_dig_prompts.txt', 'w') as f:
    for prompt in best_prompts:
        f.write(prompt + '\n')


 



