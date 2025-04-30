objects = [ 'city', 'beach', 'house', 'festival', 'road', 'dress', 'flag', 'park', 'wedding', 'kitchen']


# geo_bias_top_20 = [
#     "refrigerator",
#     "alcoholic drinks",
#     "car",
#     "bike",
#     "cooking pots",
#     "freezer",
#     "toilet",
#     "bathroom/toilet",
#     "washing detergent",
#     "source of heat",
#     "source of cool",
#     "toothbrush",
#     "cooking utensils",
#     "shower",
#     "menstruation pads / tampax",
#     "dishwasher",
#     "mosquito protection",
#     "fruit trees",
#     "phone",
#     "trash/waste"
# ]


regions = ['Africa', 'Asia', 'Europe', 'North America', 'South America', ]



prompts = [] 

for object in objects:
    prompts.append(f"{object}")
    for region in regions:
        prompts.append(f"{object} in {region}")
        


# Save prompts to a text file
with open('second_paper_prompts.txt', 'w') as f:
    for prompt in prompts:
        f.write(f"{prompt}\n")

print(f"Saved {len(prompts)} prompts to second_paper_prompts.txt")




