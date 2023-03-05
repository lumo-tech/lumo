from collections import defaultdict


class tree(dict):
    """Implements Perl's autovivification feature.

    This class extends Python's built-in dict class to allow for automatic creation of nested dictionaries
    on access of non-existent keys. It accomplishes this by overriding the __getitem__ method to recursively
    create new nested trees on access of a non-existent key. Additionally, the walk method is provided to
    allow for iterating over all keys and values in the tree.
    """

    def __getitem__(self, item):
        """Override __getitem__ to automatically create new trees on access of non-existent keys.

        Args:
            item: The key being accessed.

        Returns:
            The value of the key.
        """
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

    def walk(self):
        """Iterate over all keys and values in the tree.

        This method yields a tuple containing the current key and value, and recursively calls itself
        on any nested tree values.

        Yields:
            A tuple containing the current key and value.
        """
        for k, v in self.items():
            yield k, v
            if isinstance(v, tree):
                for kk, vv in v.walk():
                    yield f'{k}/{kk}', vv


class Node:
    """Represents a node in the Forest.

    Attributes:
        HEAD (int): A class variable indicating the head node.
        MID (int): A class variable indicating a mid node.
        TAIL (int): A class variable indicating the tail node.
        value (any): The value held by the node.
        link (list): A list of the node's adjacent nodes.
        stage (int): The position of the node in the linked list.
    """
    HEAD = 0
    MID = 1
    TAIL = 2

    def __init__(self):
        self.value = None
        self.link = []
        self.stage = None

    @property
    def is_head(self):
        """Returns True if the node is a head node, else False."""
        return self.stage == self.HEAD

    @property
    def is_mid(self):
        """Returns True if the node is a mid node, else False."""
        return self.stage == self.MID

    @property
    def is_tail(self):
        """Returns True if the node is a tail node, else False."""
        return self.stage == self.TAIL

    def set_stage(self, stage):
        """Sets the stage attribute of the node to the specified value.

        Args:
            stage (int): An integer indicating the position of the node in the linked list.

        Returns:
            Node: The node with the updated stage attribute.
        """
        self.stage = stage
        return self

    def set_value(self, val):
        """Sets the value attribute of the node to the specified value.

        Args:
            val (any): The value to set the node's value attribute to.

        Returns:
            Node: The node with the updated value attribute.
        """
        self.value = val
        return self

    def add_link(self, y):
        """Adds the specified node to the list of adjacent nodes.

        Args:
            y (Node): The node to add to the list of adjacent nodes.
        """
        self.link.append(y)

    def __repr__(self):
        return f'Node({self.stage} ,{len(self.link)}, {self.value})'


class Forest:
    """
    Represents a directed acyclic graph (DAG) where nodes are categorized as head, mid or tail.

    Attributes:
        dic (defaultdict): A dictionary to store the nodes of the graph.
        order (list): A list to maintain the order of the nodes.
        tail (set): A set to store the tail nodes of the graph.

    """

    def __init__(self):
        self.dic = defaultdict(Node)
        self.order = []
        self.tail = set()

    def add_head(self, x, val=None):
        """
        Adds a new head node to the graph with the given value.

        Args:
            x: The node to be added.
            val: The value associated with the node. Defaults to None.

        Returns:
            The updated Forest object.

        """
        self.dic[x].set_value(val).set_stage(Node.HEAD)
        self.order.append(x)
        return self

    def check_node_type(self, x):
        """
        Checks if a node is already present in the graph.

        Args:
            x: The node to be checked.

        Returns:
            True if the node is already present, False otherwise.

        """
        return x in self.dic

    def add_link(self, x, y, y_val=None):
        """
        Adds a new mid node to the graph and links it with an existing head node.

        Args:
            x: The head node to be linked with the new mid node.
            y: The new mid node to be added.
            y_val: The value associated with the new mid node. Defaults to None.

        Returns:
            The updated Forest object.

        Raises:
            AssertionError: If the head node is not already present in the graph or the mid node is already present.

        """
        assert x in self.dic, f'x must already existed in graph, has {self.order}, got {x}'
        assert y not in self.dic, f'y must be a new node in graph, has {self.order}, got {y}'
        self.dic[x].add_link(y)
        self.dic[y].set_value(y_val).set_stage(Node.MID)
        self.order.append(y)
        return self

    def add_tail(self, x, y, y_val=None):
        """
        Adds a new tail node to the graph and links it with an existing head or mid node.

        Args:
            x: The node to be linked with the new tail node.
            y: The new tail node to be added.
            y_val: The value associated with the new tail node. Defaults to None.

        Returns:
            The updated Forest object.

        Raises:
            AssertionError: If the head or mid node is not already present in the graph or the tail node is already present.

        """
        assert x in self.dic, f'x must already existed in graph, has {self.order}, got {x}'
        assert y not in self.dic, f'y must be a new node in graph, has {self.order}, got {y}'
        self.dic[x].add_link(y)
        self.dic[y].set_value(y_val).set_stage(Node.TAIL)
        self.order.append(y)
        self.tail.add(y)
        return self

    def __iter__(self):
        """
        Returns an iterator that iterates over the nodes in the graph in a Breadth-First-Search (BFS) order.

        Returns:
            An iterator that yields a tuple with the node and its corresponding value.

        """
        stack = []
        mem = set()

        if len(self.order) > 0:
            stack.append(self.order[0])

        while len(stack) > 0:
            key = stack.pop(0)
            if key in mem:
                continue
            yield key, self.dic[key]

            mem.add(key)
            for key in self.dic[key].link:
                stack.append(key)
