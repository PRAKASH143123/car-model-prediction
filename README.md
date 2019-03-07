## Car brand/model prediction

Check the kaggle solutions [here](https://www.kaggle.com/c/fcis-deep-learning-competition)

and the project writeup [here](https://adhaamehab.me/deep-learning/2018/12/25/car-brand-and-model-recognition-with-tensorflow.html)


### Retraing 

```shell
python retrain.py  \
--bottleneck_dir=tf_files/bottlenecks \
--how_many_training_steps=50000 \
--model_dir=models/ \
--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
 --output_graph=tf_files/retrained_graph.pb \
--output_labels=tf_files/retrained_labels.txt \
--architecture="${ARCHITECTURE}" \
--image_dir=data/train

```

### Generate test result

```shell
python test.py \ 
--graph=tf_files/retrained_graph.pb  \
--image=data/test/ \
--labels=tf_files/retrained_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result
```