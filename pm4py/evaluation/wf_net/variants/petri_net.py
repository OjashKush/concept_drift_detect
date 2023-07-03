import copy

from pm4py.objects.petri import utils as pn_utils
from pm4py.objects.petri.petrinet import PetriNet


def _short_circuit_petri_net(net):
    """
    Creates a short circuited Petri net,
    whether an unique source place and sink place are there,
    by connecting the sink with the source

    Parameters
    ---------------
    net
        Petri net

    Returns
    ---------------
    boolean
        Boolean value
    """
    s_c_net = copy.deepcopy(net)
    no_source_places = 0
    no_sink_places = 0
    sink = None
    source = None
    for place in s_c_net.places:
        if len(place.in_arcs) == 0:
            source = place
            no_source_places += 1
        if len(place.out_arcs) == 0:
            sink = place
            no_sink_places += 1
    if (sink is not None) and (source is not None) and no_source_places == 1 and no_sink_places == 1:
        # If there is one unique source and sink place, short circuit Petri Net is constructed
        t_1 = PetriNet.Transition("short_circuited_transition", "short_circuited_transition")
        s_c_net.transitions.add(t_1)
        # add arcs in short-circuited net
        pn_utils.add_arc_from_to(sink, t_1, s_c_net)
        pn_utils.add_arc_from_to(t_1, source, s_c_net)
        return s_c_net
    else:
        return None


def apply(net, parameters=None):
    """
    Checks if a Petri net is a workflow net

    Parameters
    ---------------
    net
        Petri net
    parameters
        Parameters of the algorithm

    Returns
    ---------------
    boolean
        Boolean value
    """
    if parameters is None:
        parameters = {}

    import networkx as nx

    scnet = _short_circuit_petri_net(net)
    if scnet is None:
        return False
    nodes = scnet.transitions | scnet.places
    graph = nx.DiGraph()
    while len(nodes) > 0:
        element = nodes.pop()
        graph.add_node(element.name)
        for in_arc in element.in_arcs:
            graph.add_node(in_arc.source.name)
            graph.add_edge(in_arc.source.name, element.name)
        for out_arc in element.out_arcs:
            graph.add_node(out_arc.target.name)
            graph.add_edge(element.name, out_arc.target.name)
    if nx.algorithms.components.is_strongly_connected(graph):
        return True
    else:
        return False
