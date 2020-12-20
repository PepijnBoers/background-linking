class Node:
    def __init__(self, name: str, node_type: str, locations: list, tf: int):
        self.__name = name
        self.__node_type = node_type
        self.__locations = locations
        self.__tf = tf
        self.__weight = 0.0

        self.tfidf = 0.0  # unused

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def weight(self):
        return self.__weight

    @weight.setter
    def weight(self, weight):
        self.__weight = weight

    @property
    def node_type(self):
        return self.__node_type

    @node_type.setter
    def node_type(self, node_type):
        self.__node_type = node_type

    @property
    def locations(self):
        return self.__locations

    def add_location(self, location):
        self.__locations.append(location)

    @property
    def tf(self):
        return self.__tf

    @tf.setter
    def tf(self, tf):
        self.__tf = tf

    def __str__(self):
        if self.__node_type != "term":
            name = self.__name.replace(" ", "_")
            return f"ENTITY/{name}"
        else:
            return self.__name
