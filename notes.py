'''
Issues comments:
- Segmentation fault: ulimit -s to see current value; increase it with ulimit -s <new_value>

'''


''' To-do list:
1)
- Add last bit on genome that signifies the output layer type   ### done
- Set up evaluation for:                                        ### done
    - probabilistic layer                                       ### done
    - XGboost                                                   ### done
- Add XGBoost as output layer                                   ### done
- Improve mutation/sampling with different output layers        ### done

Testing:
- save genomes/models/testing state after specified number of generations/time spend  ### done
- stop testing and save testing state with an input key         ### NOT done
- enable testing from saved point                               ### done


2)
- Add probabilistic Dense and Conv layers
- Different Conv layers/conv filers
- How to use graph to show different layer types. Current graph is focused on connection 

3) Different modules connections? Non sequential modules connection?

'''