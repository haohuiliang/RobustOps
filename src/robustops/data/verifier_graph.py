"""Original-verifier prerequisite graph, with no new database predicates."""
from itertools import product


def validate_graph(proposal,item):
    assert proposal['task_id']==item['task_id']
    assert proposal['final_success']=='original_verifiers_unchanged'
    nodes=proposal['nodes'];n=len(item['verifiers'])
    assert [node['index'] for node in nodes]==list(range(1,n+1))
    deps={node['index']:node['depends_on'] for node in nodes}
    for key,values in deps.items():
        assert len(set(values))==len(values)
        assert all(type(v) is int and v in deps and v!=key for v in values)
    def visit(key,path):
        assert key not in path,'dependency cycle'
        for child in deps[key]:visit(child,path+[key])
    for key in deps:visit(key,[])
    return deps


def progress(values,deps):
    def node(i):return int(values[i-1] and all(node(j) for j in deps[i]))
    return [node(i) for i in range(1,len(values)+1)]


def boundary_check(proposal,item):
    deps=validate_graph(proposal,item);n=len(deps);checked=0
    for values in product([0,1],repeat=n):
        refined=progress(values,deps)
        assert all(refined)==all(values),'full-success boundary changed'
        assert all(a<=b for a,b in zip(refined,values))
        checked+=1
    return dict(boolean_vectors_checked=checked,full_success_equivalent=True,
        original_verifier_count=n,edges=sum(map(len,deps.values())))




def reindex_dependencies(deps, mode_to_graph_indices):
    """Convert canonical graph indices to the unchanged verifier order of a mode."""
    n = len(deps)
    if sorted(mode_to_graph_indices) != list(range(1, n + 1)):
        raise ValueError("mode mapping must be a permutation of graph indices")
    inverse = {g: i for i, g in enumerate(mode_to_graph_indices, 1)}
    return {i: [inverse[j] for j in deps[g]]
            for i, g in enumerate(mode_to_graph_indices, 1)}
