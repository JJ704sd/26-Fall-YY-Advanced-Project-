import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polar_pipeline import parameters, spatial_edges, normalized_graph, sample_negatives, spatial_metrics, extract_features


def test_parameter_indices_and_normalization():
    d = {'FinalM11': np.ones((2, 2))}
    values = {'21':3, '31':4, '12':5, '13':12, '42':8, '43':15, '24':7, '34':24}
    d.update({'FinalM'+key: np.full((2,2), value) for key,value in values.items()})
    np.testing.assert_allclose(parameters(d,'as_stored')[0,0], [5,13,17,25])
    raw = {key:value*2 for key,value in d.items()}
    np.testing.assert_allclose(parameters(raw,'divide_by_m11'), parameters(d,'as_stored'))
    with pytest.raises(ValueError,match='confirm normalization'):
        parameters(raw,'as_stored')


def test_total_diattenuation_includes_circular_component():
    d = {'FinalM11': np.ones((2,2))}
    d.update({'FinalM'+key: np.zeros((2,2)) for key in ('12','13','14','21','31','24','34','42','43')})
    d['FinalM12'][:] = 3
    d['FinalM13'][:] = 4
    d['FinalM14'][:] = 12
    result = parameters(d, 'as_stored', include_total_diattenuation=True)
    np.testing.assert_allclose(result[0,0,[1,4]], [5,13])


def test_dgi_corruption_stays_within_roi():
    from graph_models import permute_within_graphs
    order = permute_within_graphs([4,3,5], np.random.default_rng(10))
    assert set(order[:4]) == set(range(4))
    assert set(order[4:7]) == set(range(4,7))
    assert set(order[7:]) == set(range(7,12))


def test_dgi_trains_reproducibly_on_two_disconnected_graphs(tmp_path):
    from scipy import sparse
    from graph_models import dgi_embedding
    edges = np.array([[0,1],[1,2],[2,3],[3,4]])
    graphs = [(5,edges),(5,edges)]
    graph = normalized_graph(5, edges, np.ones(len(edges)))
    adjacency = sparse.block_diag([graph,graph], format='csr')
    features = np.random.default_rng(42).normal(size=(10,8)).astype(np.float32)
    one, _ = dgi_embedding(features, adjacency, graphs, 7, 1, 'cpu', tmp_path)
    trained, _ = dgi_embedding(features, adjacency, graphs, 7, 20, 'cpu', tmp_path)
    repeated, device = dgi_embedding(features, adjacency, graphs, 7, 20, 'cpu', tmp_path)
    assert device == 'cpu' and trained.shape == (10,16)
    assert np.isfinite(trained).all()
    assert np.linalg.norm(trained-one) > 1e-4
    np.testing.assert_allclose(trained, repeated, atol=1e-7)


def test_mueller_decomposition_of_known_physical_components():
    import muellerkit
    depolarization, retardance, diattenuation = .2, .6, .3
    md = np.eye(4)
    md[0,1] = md[1,0] = diattenuation
    md[2,2] = md[3,3] = np.sqrt(1-diattenuation**2)
    mr = np.eye(4)
    mr[2,2] = mr[3,3] = np.cos(retardance)
    mr[2,3] = np.sin(retardance)
    mr[3,2] = -np.sin(retardance)
    mdep = np.diag([1,*([1-depolarization]*3)])
    matrix = mdep @ mr @ md
    assert muellerkit.physical_realizability(matrix).is_realizable
    dep, ret, dia = muellerkit.decompose_depolarizer_retarder_diattenuator(matrix)
    result = muellerkit.extract_parameters(dep, ret, dia)
    np.testing.assert_allclose(dep @ ret @ dia, matrix, atol=1e-12)
    np.testing.assert_allclose([result.total_depolarization, result.linear_retardance,
                                result.diattenuation], [depolarization,retardance,diattenuation], atol=1e-10)


def test_no_edges_across_background_and_normalized_self_loops():
    labels = np.array([[1,1,0,3],[1,2,0,3],[0,2,0,0]])
    edges, lengths = spatial_edges(labels)
    np.testing.assert_array_equal(edges, [[0,1]])
    assert lengths.tolist() == [2]
    a=normalized_graph(3,edges,np.ones(len(edges)))
    np.testing.assert_allclose(a.toarray(),[[.5,.5,0],[.5,.5,0],[0,0,1]])


def test_negative_samples_do_not_cross_rois_or_use_real_edges():
    graphs=[(4,np.array([[0,1],[1,2]])),(3,np.array([[0,1]]))]
    negatives=sample_negatives(graphs,np.random.default_rng(2))
    assert len(negatives)==3
    for a,b in negatives:
        assert a<b
        assert (b<4) or (a>=4)
        assert (a,b) not in {(0,1),(1,2),(4,5)}


def test_area_weighting_and_entropy_do_not_imply_spatial_order():
    block=np.array([[0,0,1,1],[0,0,1,1]])
    checker=np.array([[0,1,0,1],[1,0,1,0]])
    bm,bp,_=spatial_metrics(block,2)
    cm,cp,_=spatial_metrics(checker,2)
    np.testing.assert_allclose(bp,cp)
    assert bm['shannon_entropy']==pytest.approx(cm['shannon_entropy'])
    assert bm['mixed_neighbor_fraction']<cm['mixed_neighbor_fraction']
    labels=np.array([[1,1,1],[0,0,2]])
    maps=np.ones((2,3,4),dtype=np.float32)
    _,area=extract_features(maps,labels)
    np.testing.assert_array_equal(area,[3,1])
