# Project Write-Up

## Explaining Custom Layers

Custom layers are neural network model layers that are not natively supported by a given model framework.

The process behind converting custom layers involves two necessary custom layer extensions Custom Layer Extractor
(responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial) and Custom Layer Operation
(responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters)

Some of the potential reasons for handling custom layers are
- When a layer isn’t supported by the Model Optimizer ,Model Optimizer does not know about the custom layers so it needs to taken care of and also need to handle for handle unsupported layers at the time of inference.
- allow model optimizer to convert specific model to Intermediate Representation.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were...

The difference between model accuracy pre- and post-conversion was...

Accuracy of the pre-conversion model = moderate (less than post-conversion) and post-conversion model = Good
The size of the model pre- and post-conversion was...

size of the fozen inference graph(.pb file) = 69.7Mb and size of the pos-conversion model xml+bin file = 67.5Mb
The inference time of the model pre- and post-conversion was...

Inference time of the pre-conversion model:- Avg inference time:- 144.42 ms, min inference time:- 89.60 ms, max inference time:- 5954.10 ms
Inference time of the post-conversion model:- Avg inference time:- 2.68 ms, min inference time:- 0.31 ms, max inference time:- 67.52 ms
The CPU Overhead of the model pre- and post-conversion was...

cpu overhead of the pre conversion model:- Around 65% per core
cpu overhead of the post conversion model:- Around 40% per core
compare the differences in network needs and costs of using cloud services as opposed to deploying at the edge...

Edge model needs only local network connection or edge model can used with very low speed compared to cloud.
cost of the renting server at cloud is so high. where edge model can run on minimal cpu with local network connection.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are

1.  Restrictions of people movements (due to COVID-19) one of the potential uses could be control the acces at common zones in offices, bank offices, etc...

2. Use to count how many people visited for specific time (by using total count and duration).

3. Monitoring number of people ; Control of work zones where only one people must be in the zone and the other workers must be outside the work zone. For example, in zones with highly risk of explosion.

4. In retail , control of the people and the time inside the shop could be important in order to select the exposition and other issues.

5. Easily monitor a specific area

6. Intrusion detection 

Each of these use cases would be useful because it about detecting and counting people in the specific area. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
- Lighting:- the light will be important in order to obtain a good result. Lighting is most assential factor which affects to result of model. We need input image with lighting because model can't predict so accurately if input image is dark. So monitored place must have lights.
- Model accuracy:- Deployed edge model must have high accuracy because deployed edge model works in real time if we have deployed low accuracy model then it would give faulty results which is no good for end users.
- Camera focal length:- High focal length gives you focus on specific object and narrow angle image while Low focal length gives you the wider angle. Now It's totally depend upon end user's reuqirements that which type of camera is required. If end users want to monitor wider place than high focal length camera is better but model can extract less information about object's in picture so it can lower the accuracy. In compare if end users want to monitor very narrow place then they can use low focal length camera.
- Image size:- Image size totally depend upon resolution of image. If image resolution is better then size will be larger. Model can gives better output or result if image resolution is better but for higher resolution image model can take more time to gives output than less resolution image and also take more memory. If end users have more memory and also can manage with some delay for accurate result then higher resoltuion means larger image can be use.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [SSD_inception_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - The model was insufficient for the app because it does not work correctly and it was imposible to count correctly the people in the frame.
  - I tried to improve the model for the app by changing the threshold but the result don't improve.
  
- Model 2: [faster_rcnn_inception_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
  ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model/faster_rcnn_inception_v2_coco/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
    ```
  - The model  was not bad for the app, working better than the previous model.
  - I tried to improve the model for the app by changing the threshold to 0.4, doing this the model works but not very well.

- Model 3: [ssd_mobilenet_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel
  ```
  - The model was good for the app 
  
  - I tried to improve the model for the app by changing the threshold to 0.35, doing this the model works well but it misses drawing the boxes around a person at specific time of video, for significant no of consecutive frames.  Using probability threshold 0.35. Models seems to miss person wearing dark clothes

## Conclusion

After my investigation on those tree above models, I came to  the clonclusion that the very good model and  suitable accurate model was the one existing in Intermediate Representations provided by Intel® [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

Use this commad to dowload the model 

```
python3  /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o /home/workspace/model/pre_trained/intel
```

Running the app 

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/pre_trained/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

You can see the demo here 
[demo](https://youtu.be/GEbNlaXVVrA)