name_to_id = {}
id_to_name = [""]
id_counter = 1

with open("characters.txt", "r") as file:
    for line in file:
        # Remove leading and trailing whitespace and then assign an ID
        name = line.strip()
        name_to_id[name] = id_counter
        id_to_name.append(name)
        id_counter += 1
