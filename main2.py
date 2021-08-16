# import tfne

# blueprint_graph.append()
from tfne.encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprintNode
from tfne.encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprintConn

output_layers        = [{'node': 1, 'species': 1}, 
                        {'node': 2, 'species': 1}, 
                        {'conn_start': 1, 'conn_end': 1, 'enabled': True}]

# print(output_layers[2])

blueprint_graph = list()
blueprint_graph.append(CoDeepNEATBlueprintNode(gene_id=1, node=1, species=1))
blueprint_graph.append(CoDeepNEATBlueprintNode(gene_id=1, node=2, species=1))
blueprint_graph.append(CoDeepNEATBlueprintConn(gene_id=1, conn_start=2, conn_end=1, enabled=True))

print(blueprint_graph)

#             "1": {
#                 "node": 1,
#                 "species": null
#             },
#             "2": {
#                 "node": 2,
#                 "species": 1
#             },
#             "3": {
#                 "conn_start": 1,
#                 "conn_end": 2,
#                 "enabled": true
#             }

