import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
from openvino.inference_engine import IECore

cpu_extension = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

class Network:
    def __init__(self):
        self.net = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.plugin = None
        self.exec_network = None

    
    def load_model(self,model,cpu_extension, device,plugin= None):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.plugin = IECore()
            

        if not plugin:
            log.info("Initializing plugin {} ".format(device))
            self.plugin = IECore()
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in device:
            self.plugin.add_extension(cpu_extension, "CPU")
        
        # Read IR
        log.info("Reading IR...")
        
        log.info("Loading IR ")
        self.net = IENetwork(model=model_xml, weights=model_bin)
        
        
        supported_layers = self.plugin.query_network(network=self.net, device_name="CPU")
   
    #Check for any unsupported layers
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
                 
        
        self.exec_network = self.plugin.load_network(self.net, device)


        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        
        return
    
    def get_input_shape(self):
        return self.net.inputs[self.input_blob].shape
    
    def exec_net(self,request_id,image_p):
        #Start an asynchronous request ###
        
        
        self.infer_request = self.exec_network.start_async(
            request_id=request_id, inputs={self.input_blob: image_p})
        return
    
    def wait(self):
        infer_status = self.exec_network.requests[0].wait(-1)
        return infer_status
   
    def get_output(self):
        out = self.exec_network.requests[0].outputs[self.out_blob]
        return out
    
    
    def clean(self): 
        del self.net_plugin 
        del self.net