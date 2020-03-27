
import numpy as np
from numpy import pi,sin,cos,arctan
import subprocess
import sys
## quickly run all styles with default params,
## compile a matrix

#########################################

args={}
assert len(sys.argv)>2,"usage: python style_blast.py <path/to/content>"
content=sys.argv[1]
args['content_image']=str(content)
### Commonly changed for art
args['output_image']="out/out.jpg" #[out.png]
args['style_image']="styles/elephant.jpg" #Style target image [examples/inputs/seated-nude.jpg]
args['style_scale']=1 #[1]
args['style_weight']=100 #[100]
args['content_weight']=5 #[5]
args['init']="random" #random|image [random]
#args['init_image']="" #[]
args['original_colors']=0 #[0]
args['image_size']=512 #Maximum height / width of generated image [512]
args['num_iterations']=701 #[1000]
args['save_iter']=100 #[100]
args['print_iter']=50 #[50]

### I havent touched
args['tv_weight']=0.001 #[0.001]
args['pooling']="max" #max|avg [max]
args['seed']=-1 #[-1]
args['proto_file']="models/VGG_ILSVRC_19_layers_deploy.prototxt" #[models/VGG_ILSVRC_19_layers_deploy.prototxt]
args['model_file']="models/VGG_ILSVRC_19_layers.caffemodel" #[models/VGG_ILSVRC_19_layers.caffemodel]
args['content_layers']="relu4_2" #layers for content [relu4_2]
args['style_layers']="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1" #layers for style [relu1_1,relu2_1,relu3_1,relu4_1,relu5_1]
args['lbfgs_num_correction']=0 #[0]
#args['style_blend_weights']="" #[nil]

### Seem to be well tuned
args['optimizer']="lbfgs" #lbfgs|adam [lbfgs]
#args['normalize_gradients']="" #[]
args['learning_rate']=10 #[10]
args['gpu']=0 #Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1 [0]
args['cudnn_autotune']="" #[false]
args['backend']="cudnn" #nn|cudnn|clnn [nn]

def build_args(args):
    return list(filter(None,["th", "neural_style.lua"]+ sum([ "-{} {}".format(key,args[key]).split(" ") for key in args.keys() ], []) ))
def get_name(path):
    return path.split("/")[-1].split(".")[0]
#########################################


content=args['content_image'].split("/")[-1].split(".")[0]
suff=content+"-default.jpg"

rowlist=[]
for style in filter(None, subprocess.run("ls styles/*.jpg",shell=True,capture_output=True).stdout.decode('utf-8').split("\n")  ):
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    print(style)
    style=get_name(style)+".jpg"
    args['style_image']="styles/"+style #Style target image [examples/inputs/seated-nude.jpg]


    # build output filename
    args['output_image']="output/"+style.rstrip(".jpg")+"-"+suff

    #run neural_style
    print(build_args(args))
    subprocess.run(build_args(args) )

    #Append the interation outputs into a row, keep track of the file name
    rowname="{}-row.jpg".format(args['output_image'].rstrip(".jpg") )
    subprocess.run(\
        ["convert", "+append", "{0}_*.jpg".format( args['output_image'].rstrip(".jpg")   ), rowname]\
        )
    rowlist.append(rowname)
    subprocess.run("notify-send -t 4000 'finished another row!' ",shell=True)


subprocess.run(\
            ["convert", "-append"]+rowlist+["output/style-blast-{}".format(suff)]\
        )

