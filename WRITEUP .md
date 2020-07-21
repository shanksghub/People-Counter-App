# Project Write-Up


## Explaining Custom Layers


Some of the potential reasons for handling custom layers are...

To add different pieces of code.

I have referred to various sources for OpenVino for the reasons for Handling Custom layers - 

Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation.


Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

If your topology contains layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error.

Custom layers are not recognised and need to be added separately


The process behind converting custom layer involves -

When implementing a custom layer for your pre-trained model in the Intel® Distribution of OpenVINO™ toolkit, you will need to add extensions to both the Model Optimizer and the Inference Engine.

CPU extension used in the project  - /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so


For more about processes behind converting custom layers one can visit https://docs.openvinotoolkit.org/latest/openvino_docs_HOWTO_Custom_Layers_Guide.html and https://software.intel.com/content/www/us/en/develop/articles/openvino-custom-layers-support-in-inference-engine.html.



Custom Layer Extensions for the Model Optimizer
The following figure shows the basic processing steps for the Model Optimizer highlighting the two necessary custom layer extensions, the Custom Layer Extractor and the Custom Layer Operation.

MO_extensions_flow.png
https://r953259c960231xjupyterlcm2lsxnf.udacity-student-workspaces.com/lab/tree/MO_extensions_flow.png

The Model Optimizer first extracts information from the input model which includes the topology of the model layers along with parameters, input and output format, etc., for each layer. The model is then optimized from the various known characteristics of the layers, interconnects, and data flow which partly comes from the layer operation providing details including the shape of the output for each layer. Finally, the optimized model is output to the model IR files needed by the Inference Engine to run the model.

The Model Optimizer starts with a library of known extractors and operations for each supported model framework which must be extended to use each unknown custom layer. The custom layer extensions needed by the Model Optimizer are:

Custom Layer Extractor
Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial.



Custom Layer Operation
Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters.
The --mo-op command-line argument shown in the examples below generates a custom layer operation for the Model Optimizer.
Custom Layer Extensions for the Inference Engine
The following figure shows the basic flow for the Inference Engine highlighting two custom layer extensions for the CPU and GPU Plugins, the Custom Layer CPU extension and the Custom Layer GPU Extension.

IE_extensions_flow.png
https://r953259c960231xjupyterlcm2lsxnf.udacity-student-workspaces.com/lab/tree/IE_extensions_flow.png


Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device:

Custom Layer CPU Extension
A compiled shared library (.so or .dll binary) needed by the CPU Plugin for executing the custom layer on the CPU.
Custom Layer GPU Extension
OpenCL source code (.cl) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (.xml) needed by the GPU Plugin for the custom layer kernel.

Here is a link to Custom Layer guide - https://docs.openvinotoolkit.org/latest/openvino_docs_HOWTO_Custom_Layers_Guide.html#Tensorflow-models-with-custom-layers

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...accuracy

The difference between model accuracy pre- and post-conversion was... Better accuracy post conversion than preconversion.

The size of the model pre- and post-conversion was...66.5 MB and 64.5 MB.

The inference time of the model pre- and post-conversion was 69.855 for 0.5 prob threshold and 71.872 for 0.2 prob threshold.


What about differences in network needs and costs of using cloud services as opposed to at the edge?
As opposed to the Edge costs would be very high if we use cloud services and also using such services in remote areas or where internet is not good would not be feasible.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

1. During lockdown to prevent more people gathering at a hotspot.
2. To provide data to security to keep crowd under control.
3.  To help guide people if there is overcrowding.
4. To not let wrong elements enter a place.

Each of these use cases would be useful because...
1. It can beat human eye for counting large number of people.
2. Without human intervention or with less of it, crowd can be controlled.
3. For security reasons, places where security would take time to reach can be observed and data can be collected.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

Lighting - 
There should be atleast moderate lighting so that we get good accuracy.If it is too dark then we would need more advanced software and hardware.

Model accuracy - 
We need models with good accuracy and algorithms as there could be misinformation and wrong data analytics if these parameters are not good.

Focal Length and Image Size
The focal length and image size would depend on case to case and depending where we are installing such systems.




## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I used 
SSD MobileNet V2 COCO

I used 
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

to download the model SSD MobileNet V2 COCO model from here. Used the tar -xvf command with the downloaded file to unpack it :
-  tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz                   
I converted the SSD MobileNet V2 model from TensorFlow using this command - 

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

and got 
  
 frozen_inference_graph.xml
 frozen_inference_graph.bin
 
 After sourcing env using this command - source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
 
 to run the code I used -
 
 python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
 
 
 
 
 