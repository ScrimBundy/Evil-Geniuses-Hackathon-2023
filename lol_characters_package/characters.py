class Character:

    def __init__(self, name: str, id: int, top: bool,
                 jungle: bool, mid: bool, bot: bool,
                 support: bool, unused: bool) -> None:
        self.name = name
        self.id = id
        self.top = top
        self.jungle = jungle
        self.mid = mid
        self.bot = bot
        self.support = support
        self.unused = unused


class Characters:
    def __init__(self, character_file):
        self.character_list = []
        self.top_characters = []
        self.jungle_characters = []
        self.mid_characters = []
        self.bot_characters = []
        self.support_characters = []
        self._read_characters_from_file(character_file)

    def _read_characters_from_file(self, character_file):
        """
        Initialization function that pulls data from a file called characters.csv.
        The first row is column names.
        The columns in order are:
            Name, Top, Jungle, Middle, Bottom, Support, Unused.
        Each contains a zero if the character is not in that category, a 1 if both
        Riot and op.gg agree on the assignment, 2 if assigned by op.gg, and 3 if
        assigned by Riot.
        :return: None
        """
        id_counter = 0

        with open(character_file, "r", encoding="utf-8-sig") as file:
            content = file.read()
            lines = content.split("\n")
            for line in lines[1:]:
                # Ignore empty lines
                if line == "": continue
                # Remove leading and trailing whitespace
                line = line.strip("\r\n\t ")
                # Tokenize using comma seperator
                tokens = line.split(",")

                # Define Character values
                name = tokens[0]
                id = id_counter
                top = tokens[1] != 0  # True/False
                jungle = tokens[2] != 0  # True/False
                mid = tokens[3] != 0  # True/False
                bot = tokens[4] != 0  # True/False
                support = tokens[5] != 0  # True/False
                unused = tokens[6] != 0  # True/False

                # Create Character object
                char = Character(name, id, top, jungle, mid, bot, support, unused)

                # Append Character to appropriate lists
                self.character_list.append(char)
                if top:
                    self.top_characters.append(char)
                if jungle:
                    self.jungle_characters.append(char)
                if mid:
                    self.mid_characters.append(char)
                if bot:
                    self.bot_characters.append(char)
                if support:
                    self.support_characters.append(char)

                id_counter += 1
