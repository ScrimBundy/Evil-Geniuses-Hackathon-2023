from lol_characters_package.characters import Characters


def test():
    chars = Characters()
    chars.read_characters_from_file()
    for char in chars.character_list:
        print(char.name)



if __name__ == "__main__":
    test()