import os
import pandas
import numpy as np
import networkx as nx

from collections import defaultdict


def stripnodename(df):
    '''inplace change node_id fields into int id fields    
    '''
    df['node_id1'] = df['node_id1'].apply(lambda x: int(x[5:]))
    df['node_id2'] = df['node_id2'].apply(lambda x: int(x[5:]))


def removeweakclasses(df, weaklabels, debug=True, share_to_keep=None):
    if share_to_keep is None:
        for c in weaklabels:
            if debug:
                print("dropping", c)
            df.drop(df[df['label_id1'] == c].index, inplace=True)
            df.drop(df[df['label_id2'] == c].index, inplace=True)
    else:
        # keep some of the labels
        # 1. find how many nodes of the label is present
        
        df1 = df[['node_id1', 'label_id1']].rename(columns={'node_id1': 'node_id', 'label_id1': 'label_id'})
        df2 = df[['node_id2', 'label_id2']].rename(columns={'node_id2': 'node_id', 'label_id2': 'label_id'})
        uniqnodes = pandas.concat([df1, df2]).drop_duplicates()
        
        for c in weaklabels:
            weaknodes = uniqnodes[uniqnodes['label_id']==c]['node_id'].to_numpy()
            print('label to remove:', c)            
            intshare_to_remove = int(weaknodes.shape[0]*(1-share_to_keep))
            print(f"total: {weaknodes.shape[0]}, removing: {intshare_to_remove}")
            print("removing nodes:", weaknodes[0:intshare_to_remove])
            df.drop(df[df['node_id1'].isin(weaknodes[0:intshare_to_remove])].index, inplace=True)
            df.drop(df[df['node_id2'].isin(weaknodes[0:intshare_to_remove])].index, inplace=True)


def getuniqnodes(df, dftype, debug=True):
    df1 = \
        df[['node_id1', 'label_id1']].rename(columns={'node_id1': 'node_id', 'label_id1': 'label_id'})
    df2 = \
        df[['node_id2', 'label_id2']].rename(columns={'node_id2': 'node_id', 'label_id2': 'label_id'})
    uniqnodes = pandas.concat([df1, df2]).drop_duplicates()
    nodecount = uniqnodes.shape[0]
    uniqids = uniqnodes.drop_duplicates('node_id')
    idcount = uniqids.shape[0]
    if debug:
        if nodecount != idcount:
            print(f"WARNING! inconsistent labels in {dftype} datafile")
        print(f"Unique ids in {dftype} datafile: {idcount}")
    return uniqids


def checkuniqids(uniqids):
    idints = uniqids['node_id']
    if min(idints) > 0:
        print("WARNING: ids are starting from", min(idints), "must be translated!")
    else:
        print("OK: Ids are starting from 0")
    if max(idints) - min(idints) + 1 == idints.shape[0]:
        print("OK: Ids are consecutive")
    else:
        print("WARNING: ids are not consequtive, must be translated!")

        
def removecloserelatives(df, maxweight):
    df_filtered = df[df["ibd_sum"]>maxweight] 
    uniqids = getuniqnodes(df_filtered, "relatives removal")
    nodes = uniqids["node_id"].to_numpy()
    counts = np.ones(nodes.shape[0], dtype=np.int64)

    df1 = df_filtered[['node_id1']].rename(columns={'node_id1': 'node_id'})
    df2 = df_filtered[['node_id2']].rename(columns={'node_id2': 'node_id'})
    repeatingnodes = pandas.concat([df1, df2])

    for idx in range(nodes.shape[0]):
        node = nodes[idx]
        occurences = repeatingnodes[repeatingnodes["node_id"]==node].shape[0]
        counts[idx] = occurences
    
    indices = counts.argsort()[::-1]
    descendingpower = nodes[indices]
    
    remover = df_filtered.copy()
    to_remove = []
    for idx,node in enumerate(descendingpower):
        tmp = remover[ ((remover.node_id1 != node) & ( remover.node_id2 != node) )]
        if tmp.shape[0] < remover.shape[0]:
            to_remove.append(node)
            remover = tmp
        #else:
        #    print("for node", idx, "no rows removed", remover.shape[0], "left")    
        if remover.shape[0]==0:
            break
    print("removing", len(to_remove), "nodes as their close relatives are present")
    # closerelativesdf = df[df['node_id1'].isin(to_remove) and df['node_id2'].isin(to_remove)].copy()
    df.drop(df[df['node_id1'].isin(to_remove)].index, inplace=True)
    df.drop(df[df['node_id2'].isin(to_remove)].index, inplace=True)
    return to_remove
    
        

def load_pure(datafilename, labelfilenames={}, unknown_share=0, minclassize=None, removeclasses=None, only=None,
              maxweight=None, include_unknown=None, debug=True):
    '''Verify and load files from dataset1 pure format
        into numpy arrays
    
    Parameters
    ----------
    datafilename: str
        filename for the list of edges with weights, counts and node descriptions         
    labelfilenames: dict
        use when there are labels in separate files, not in the graph datafile. format: {"label":"labeldatafile.csv"}
    unknown_share: float
        if there are unlabelled nodes include some of them (and their edges) with label "masked"
    minclassize: int
        minimum class size to be included to returned nodes 
    removeclasses: list
        list of labels to remove from dataset
    Returns
    -------
    pairs: ndarray 
        array of N x 3 ints for ibd counts
        each row is of the form [node1, node2, number of ibd]
    weights: ndarray
        array of M floats of weights 
        k-th weight corresponds to the k-th row of the pairs
        if ibd_max column is present, then also contains a column for max values
    labels: ndarray
        array of ints with class number for every node
    labeldict: dict of labels eg {"pop1":0, "pop2":1}
    idxtranslator: i-th node_id in the i-th element
    '''
    
    
    dfibd = pandas.read_csv(datafilename)
    if labelfilenames == {}:
        stripnodename(dfibd)
    else:
        # TODO remove extra work as we already have normal structure in this case
        
        # add columns with label_id1 and label_id2 with numerical stirngs
        # first fill with masked then use all the label datafiles
        dfibd["label_id1"] = 'masked'
        dfibd["label_id2"] = 'masked'

        labelarrays = {}
        for lbl in labelfilenames:
            onelabeldf = pandas.read_csv(labelfilenames[lbl])
            ids = np.ravel(onelabeldf[['anonymized_id']].to_numpy())
            labelarrays[lbl] = ids
            # print(ids)
            # print(type(ids))
            print("label", lbl, "size", ids.shape)
            print("finding unique indices by file")
            # 1. one id multiple times in one file -> just inform
            ids, counts = np.unique(ids, return_counts=True)
            multips = counts > 1
            print(list(zip(ids[multips], counts[multips])))

        # 2. one id appears in multiple datasets -> inform and remove them
        to_remove = set()
        for lbl1 in labelarrays:
            for lbl2 in labelarrays:
                if lbl1 != lbl2:
                    isect = set(labelarrays[lbl1]).intersection(labelarrays[lbl2])
                    print(f"Nodes in both {lbl1} and {lbl2}:")
                    print(isect)
                    to_remove.update(isect)
        # remove 
        print("removing multi-labelled nodes:")
        for lbl in labelarrays:
            len_before = labelarrays[lbl].shape[0]
            labelarrays[lbl] = np.array(list(set(labelarrays[lbl]).difference(to_remove)))
            len_after = labelarrays[lbl].shape[0]            
            print(f"{lbl}: {len_before}->{len_after}")
            # print(labelarrays[lbl])
        dfibd.drop(dfibd[dfibd['node_id1'].isin(to_remove)].index, inplace=True)
        dfibd.drop(dfibd[dfibd['node_id2'].isin(to_remove)].index, inplace=True)
        # TODO
        # 4. check indices are present in graph -> inform and remove non-present
        
        # 3. after graph loading find close relatives with different labels and inform
        
        for lbl in labelarrays:
            print ("fixing label", lbl)
            dfibd.loc[dfibd["node_id1"].isin(labelarrays[lbl]), "label_id1"] = lbl
            dfibd.loc[dfibd["node_id2"].isin(labelarrays[lbl]), "label_id2"] = lbl
            #for itr, idx in enumerate(labelarrays[lbl]):
                #print("fixing", lbl, " : ", itr, "of ", labelarrays[lbl].shape[0], "elements")
                #dfibd.loc[dfibd["node_id1"] == idx, "label_id1"] = lbl
                #dfibd.loc[dfibd["node_id2"] == idx, "label_id2"] = lbl
        # todo we want full graph or at least unknown_share
        removeweakclasses(dfibd, ['masked'], debug=debug, share_to_keep=include_unknown)
        
        # now it is time to remove too close relatives (ibdsum>maxweight)
        # try to remove as little nodes as possible
        if not maxweight is None:
            removedcloserelatives = removecloserelatives(dfibd, maxweight)
            print("label count after removing close relatives:")
            for lbl in labelarrays:
                len_before = labelarrays[lbl].shape[0]
                labelarrays[lbl] = np.array(list(set(labelarrays[lbl]).difference(removedcloserelatives)))
                len_after = labelarrays[lbl].shape[0]            
                print(f"{lbl}: {len_before}->{len_after}")

            
            

    if not (removeclasses is None):
        removeweakclasses(dfibd, removeclasses, debug)
    uniqids = getuniqnodes(dfibd, 'ibd', debug)
    if debug:
        checkuniqids(uniqids)
    uniqids = uniqids.sort_values(by=['node_id'])

    labeldf = uniqids[['label_id']].drop_duplicates()

    # compile label dictionary
    lbl = 0
    labeldict = {}
    for _, row in labeldf.iterrows():
        labeldict[row['label_id']] = lbl
        lbl += 1

    if not (minclassize is None):
        if debug:
            print("Filter out all classes smaller than ", minclassize)
        # count and filter out rare label_id
        powerlabels = []
        powerlabelcount = []
        weaklabels = []
        weaklabelcount = []
        for c in labeldict:
            count = len(uniqids[uniqids['label_id'] == c])
            if count < minclassize:
                weaklabels.append(c)
                weaklabelcount.append(count)
            else:
                powerlabels.append(c)
                powerlabelcount.append(count)
        weakargs = np.argsort(weaklabelcount)
        powerargs = np.argsort(powerlabelcount)
        if debug:
            print("Removing following classes:")
        totalweak = 0
        for idx in range(len(weakargs)):
            sids = weakargs[idx]
            totalweak += weaklabelcount[sids]
            if debug:
                print(weaklabels[sids], weaklabelcount[sids])
        if debug:
            print("Total", totalweak, "removed")
            print("Remaining classes:")
        for idx in range(len(powerargs)):
            sids = powerargs[idx]
            if debug:
                print(powerlabels[sids], powerlabelcount[sids])

        # remove rare from uniqids
        removeweakclasses(dfibd, weaklabels, debug)

        uniqids = getuniqnodes(dfibd, 'ibd', debug)
        if debug:
            checkuniqids(uniqids)
        uniqids = uniqids.sort_values(by=['node_id'])

        labeldf = uniqids[['label_id']].drop_duplicates()

        # compile label dictionary
        lbl = 0
        labeldict = {}
        for idx in powerargs[::-1]:
            labeldict[powerlabels[idx]] = lbl
            lbl += 1

    if debug:
        print("Label dictionary:", labeldict)
    # create labels array
    labels = uniqids['label_id'].apply(lambda x: labeldict[x]).to_numpy()
    idxtranslator = uniqids['node_id'].to_numpy()
    # id translator[idx] contains idx's node name
    # it equals to idx if nodes are named from 0 consistently

    dfnodepairs = dfibd[['node_id1', 'node_id2']]
    paircount = dfnodepairs.shape[0]
    uniqpaircount = dfnodepairs.drop_duplicates().shape[0]
    if debug:
        print(f"paircount: {paircount}, unique: {uniqpaircount}")
        if paircount != uniqpaircount:
            print("Not OK: duplicate pairs")
        else:
            print("OK: all pairs are unique")

    pairs = dfibd[['node_id1', 'node_id2', 'ibd_n']].to_numpy()
    # todo order pairs
    for idx in range(pairs.shape[0]):
        n1 = pairs[idx, 0]
        n2 = pairs[idx, 1]
        if n1 > n2:
            pairs[idx, 0] = n2
            pairs[idx, 1] = n1

    if 'ibd_max' in dfibd.columns:
        weights = dfibd[['ibd_sum', 'ibd_max']].to_numpy()
    else:
        weights = dfibd[['ibd_sum']].to_numpy()

    return pairs, weights, labels, labeldict, idxtranslator


def translate_indices(pairs, idxtranslator):
    """
        replaces nonconsecutive indices from pairs to consecutive according to idxtranslator
    
    """
    result = np.empty_like(pairs)
    for idx in range(pairs.shape[0]):
        result[idx, 0] = np.where(idxtranslator == pairs[idx, 0])[0][0]
        result[idx, 1] = np.where(idxtranslator == pairs[idx, 1])[0][0]
        result[idx, 2] = pairs[idx, 2]
    return result


def getcloserelatives_w(weightedpairs, weights, threshold):
    widelinks = weights > threshold
    print("number of wide links:", np.sum(widelinks))
    relatives = []
    for pair in weightedpairs[widelinks]:
        relatives.append(pair[0])
        relatives.append(pair[1])
    relatives = list(set(relatives))
    print("total close relatives (unique):", len(relatives))
    return relatives


def fill_bins(rates, binbounds):
    '''
        returns bins for cr rates according to bin bounds
    '''
    bins = [[] for _ in range(len(binbounds) - 1)]
    for nodeidx, rate in enumerate(rates):
        for idx in range(len(bins)):
            if (binbounds[idx] <= rate) and (rate < binbounds[idx + 1]):
                bins[idx].append(nodeidx)
    return bins


def get_bin_idx(rate, binbounds):
    '''
        returns bin index for specified cr rate
    '''
    for idx in range(len(binbounds) - 1):
        if (binbounds[idx] <= rate) and (rate < binbounds[idx + 1]):
            return idx
    return -1


def get_stat_for_node_vs_group(node, bn, Gw):
    '''
        returns number of links between node and a group in graph Gw
    '''
    weights = []
    for edge in nx.edge_boundary(Gw, bn, [node], data=True):
        _, _, d = edge
        weights.append(d["weight"])
    if len(weights) > 0:
        weights = np.array(weights)
        return weights.shape[0], np.mean(weights), np.std(weights)
    else:
        return 0, 0, 0


def get_weighted_dataloaders():
    pass


def load_region_df(region_df_path):
    """
        returns fixed region dataset as pandas.DataFrame.
    """
    print(f"Loading {region_df_path}.")
    nodes_df = pandas.read_csv(region_df_path)
    print("The following values were encountered in the field 'value':")
    value_counts = nodes_df['value'].value_counts()
    print(value_counts.to_string())

    total_rows = nodes_df.shape[0]
    unique_ids = nodes_df['anonymized_id'].nunique()
    print("Total rows:", total_rows)
    print("Unique ids:", unique_ids)

    # Replace extra words in value column
    nodes_df['value'] = nodes_df['value'].str.replace(' область', '', regex=False)
    nodes_df['value'] = nodes_df['value'].str.replace(' край', '', regex=False)
    # Delete duplicate rows
    node_df = nodes_df.drop_duplicates()

    return node_df


def data_stats(datadir):
    '''
        analyzes the dataset of a graph and outputs various statistics.
    '''

    regions = ['kaluga', 'krasnodar', 'ryazan', 'vologda', 'yaroslavl']

    graph_path = os.path.join(datadir, 'df_anonymized.csv')
    kaluga_path = os.path.join(datadir, 'kaluzskaya_anonymized.csv')
    krasnodar_path = os.path.join(datadir, 'krasnodarskiy_anonymized.csv')
    ryazan_path = os.path.join(datadir, 'ryazanskaya_anonymized.csv')
    vologda_path = os.path.join(datadir, 'vologodskaya_anonymized.csv')
    yaroslavl_path = os.path.join(datadir, 'yaroslavskaya_anonymized.csv')

    print("------------------------------------------------------------")
    kaluga_df = load_region_df(kaluga_path)
    print("------------------------------------------------------------")
    krasnodar_df = load_region_df(krasnodar_path)
    print("------------------------------------------------------------")
    ryazan_df = load_region_df(ryazan_path)
    print("------------------------------------------------------------")
    vologda_df = load_region_df(vologda_path)
    print("------------------------------------------------------------")
    yaroslavl_df = load_region_df(yaroslavl_path)
    print("------------------------------------------------------------")

    kaluga_nodes = kaluga_df['anonymized_id'].unique()
    krasnodar_nodes = krasnodar_df['anonymized_id'].unique()
    ryazan_nodes = ryazan_df['anonymized_id'].unique()
    vologda_nodes = vologda_df['anonymized_id'].unique()
    yaroslavl_nodes = yaroslavl_df['anonymized_id'].unique()
    all_nodes = np.concatenate([kaluga_nodes, krasnodar_nodes, ryazan_nodes, vologda_nodes, yaroslavl_nodes])

    print(f"Total ids encountered: {len(all_nodes)}, {len(np.unique(all_nodes))} of which are unique.")

    print("------------------------------------------------------------")

    graph_df = pandas.read_csv(graph_path, dtype={
        'node_id1': 'int',
        'node_id2': 'int',
        'ibd_sum': 'float',
        'ibd_n': 'int8'
    })

    edges_df = graph_df[['node_id1', 'node_id2']]

    edges_count = len(edges_df)
    unique_edges_count = len(edges_df.drop_duplicates())

    # Check if the dataset has duplicate edges
    if edges_count != unique_edges_count:
        print('Not OK: duplicate pair')
    else:
        print('OK: all pairs are unique')

    # Combine two columns and get unique ids as a set
    nodes_in_graph = set(pandas.concat([graph_df['node_id1'], graph_df['node_id2']]).unique())
    # Get all ids from region datasets
    nodes_in_regions = set(np.unique(all_nodes))

    # Check if the dataset includes every region data id
    if nodes_in_regions.issubset(nodes_in_graph):
        print('OK: all ids are present in the dataset')
    else:
        print('Not OK: some ids are not in the dataset')

    print("------------------------------------------------------------")

    # Prints a repeated anonymized_id with the location found.
    nodes_locations = defaultdict(set)
    node_arrays = [kaluga_nodes, krasnodar_nodes, ryazan_nodes, vologda_nodes, yaroslavl_nodes]
    for i, node_array in enumerate(node_arrays):
        for node in node_array:
            nodes_locations[node].add(regions[i])

    for node, locations in nodes_locations.items():
        if len(locations) >= 2:
            print(f"Anonymized_id {node} is in {', '.join(map(str, locations))}")

    print("------------------------------------------------------------")
    print('Statistics:')
    print(f'Number of pairs in the dataset: {edges_count}')
    print(f'Number of ids in the dataset: {len(nodes_in_graph)}')
    print(f'Number of ids in the region datasets: {len(nodes_in_regions)}')

    columns_to_describe = ['ibd_sum', 'ibd_n']
    print(graph_df[columns_to_describe].describe())


if __name__ == '__main__':
    pass
