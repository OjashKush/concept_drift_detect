from collections import Counter


class Marking(Counter):
    pass

    def __hash__(self):
        r = 0
        for p in self.items():
            r += 31 * hash(p[0]) * p[1]
        return r

    def __eq__(self, other):
        if not self.keys() == other.keys():
            return False
        for p in self.keys():
            if other.get(p) != self.get(p):
                return False
        return True

    def __le__(self, other):
        if not self.keys() <= other.keys():
            return False
        for p in self.keys():
            if other.get(p) < self.get(p):
                return False
        return True

    def __add__(self, other):
        m = Marking()
        for p in self.items():
            m[p[0]] = p[1]
        for p in other.items():
            m[p[0]] += p[1]
        return m

    def __sub__(self, other):
        m = Marking()
        for p in self.items():
            m[p[0]] = p[1]
        for p in other.items():
            m[p[0]] -= p[1]
            if m[p[0]] == 0:
                del m[p[0]]
        return m

    def __repr__(self):
        # return str([str(p.name) + ":" + str(self.get(p)) for p in self.keys()])
        # The previous representation had a bug, it took into account the order of the places with tokens
        return str([str(p.name) + ":" + str(self.get(p)) for p in sorted(list(self.keys()), key=lambda x: x.name)])


class PetriNet(object):
    class Place(object):

        def __init__(self, name, in_arcs=None, out_arcs=None, properties=None):
            self.__name = name
            self.__in_arcs = set() if in_arcs is None else in_arcs
            self.__out_arcs = set() if out_arcs is None else out_arcs
            self.__properties = dict() if properties is None else properties

        def __set_name(self, name):
            self.__name = name

        def __get_name(self):
            return self.__name

        def __get_out_arcs(self):
            return self.__out_arcs

        def __get_in_arcs(self):
            return self.__in_arcs

        def __get_properties(self):
            return self.__properties

        def __repr__(self):
            return str(self.name)

        name = property(__get_name, __set_name)
        in_arcs = property(__get_in_arcs)
        out_arcs = property(__get_out_arcs)
        properties = property(__get_properties)

    class Transition(object):

        def __init__(self, name, label=None, in_arcs=None, out_arcs=None, properties=None):
            self.__name = name
            self.__label = None if label is None else label
            self.__in_arcs = set() if in_arcs is None else in_arcs
            self.__out_arcs = set() if out_arcs is None else out_arcs
            self.__properties = dict() if properties is None else properties

        def __set_name(self, name):
            self.__name = name

        def __get_name(self):
            return self.__name

        def __set_label(self, label):
            self.__label = label

        def __get_label(self):
            return self.__label

        def __get_out_arcs(self):
            return self.__out_arcs

        def __get_in_arcs(self):
            return self.__in_arcs

        def __get_properties(self):
            return self.__properties

        def __repr__(self):
            if self.label is None:
                return str(self.name)
            else:
                return str(self.label)

        name = property(__get_name, __set_name)
        label = property(__get_label, __set_label)
        in_arcs = property(__get_in_arcs)
        out_arcs = property(__get_out_arcs)
        properties = property(__get_properties)

    class Arc(object):

        def __init__(self, source, target, weight=1, properties=None):
            if type(source) is type(target):
                raise Exception('Petri nets are bipartite graphs!')
            self.__source = source
            self.__target = target
            self.__weight = weight
            self.__properties = dict() if properties is None else properties

        def __get_source(self):
            return self.__source

        def __get_target(self):
            return self.__target

        def __set_weight(self, weight):
            self.__weight = weight

        def __get_weight(self):
            return self.__weight

        def __get_properties(self):
            return self.__properties

        def __repr__(self):
            if type(self.source) is PetriNet.Transition:
                if self.source.label:
                    return "(t)" + str(self.source.label) + "->" + "(p)" + str(self.target.name)
                else:
                    return "(t)" + str(self.source.name) + "->" + "(p)" + str(self.target.name)
            if type(self.target) is PetriNet.Transition:
                if self.target.label:
                    return "(p)" + str(self.source.name) + "->" + "(t)" + str(self.target.label)
                else:
                    return "(p)" + str(self.source.name) + "->" + "(t)" + str(self.target.name)

        source = property(__get_source)
        target = property(__get_target)
        weight = property(__get_weight, __set_weight)
        properties = property(__get_properties)

    def __init__(self, name=None, places=None, transitions=None, arcs=None, properties=None):
        self.__name = "" if name is None else name
        self.__places = set() if places is None else places
        self.__transitions = set() if transitions is None else transitions
        self.__arcs = set() if arcs is None else arcs
        self.__properties = dict() if properties is None else properties

    def __get_name(self):
        return self.__name

    def __set_name(self, name):
        self.__name = name

    def __get_places(self):
        return self.__places

    def __get_transitions(self):
        return self.__transitions

    def __get_arcs(self):
        return self.__arcs

    def __get_properties(self):
        return self.__properties

    name = property(__get_name, __set_name)
    places = property(__get_places)
    transitions = property(__get_transitions)
    arcs = property(__get_arcs)
    properties = property(__get_properties)
