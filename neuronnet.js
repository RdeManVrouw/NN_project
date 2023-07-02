Math.sq = function (x){
  return x * x;
}

Math.sigmoid = function (x){
  return 1.0 / (1 + Math.exp(-x));
}
Math.relu = function (x){
  return (x < 0) ? 0.0 : x;
}
Math.capped_relu = function (x){
  return (x < 0) ? 0.0 : Math.min(x, 1);
}
Math.leaky_relu = function (x){
  return (x < 0) ? 0.1 * x : x;
}
Math.elu = function (x){
  return (x < 0) ? 0.1 * (Math.exp(x) - 1) : x;
}
Math.identity = function (x){
  return x;
}

Math.diff_sigmoid = function (x){
  var s = Math.sigmoid(x);
  return s * (1 - s);
}
Math.diff_relu = function (x){
  return (x < 0) ? 0.0 : 1.0;
}
Math.diff_capped_relu = function (x){
  return (x > 0 && x < 1) ? 1 : 0;
}
Math.diff_leaky_relu = function (x){
  return (x < 0) ? 0.1 : 1.0;
}
Math.diff_elu = function (x){
  return (x < 0) ? 0.1 * Math.exp(x) : 1;
}
Math.diff_tanh = function (x){
  return 1 - Math.sq(Math.tanh(x));
}
Math.diff_identity = function (x){
  return 1;
}

function intToString(n){
  var str = "";
  for (var i = 0; i < 4; i++){
    str = String.fromCharCode(n & 0xff) + str;
    n >>= 8;
  }
  return str;
}
function stringToInt(str){
  var n = 0;
  for (var i = 0; i < 4; i++){
    n = (n << 8) + str.charCodeAt(i);
  }
  return n;
}
const buffer = new ArrayBuffer(4);
const intView = new Int32Array(buffer);
const floatView = new Float32Array(buffer);
function floatToString(x){
  floatView[0] = x;
  return intToString(intView[0]);
}
function stringToFloat(str){
  intView[0] = stringToInt(str);
  return floatView[0];
}

class NeuronModel {
  constructor(dim_input){
    this.layer0 = [];
    for (var i = 0; i < dim_input; i++) this.layer0[i] = new InputNeuron();
    this.layers = [];
  }

  addLayer(dim, activation_function = "tanh"){
    var l;
    this.layers.push(l = new Layer(dim, activation_function));
    if (this.layers.length > 1){
      l.connectFull(this.layers[this.layers.length - 2]);
    } else {
      for (var i = 0; i < l.neurons.length; i++){
        for (var j = 0; j < this.layer0.length; j++) l.neurons[i].addChild(this.layer0[j]);
      }
    }
  }

  predict(input){
    for (var i = 0; i < this.layer0.length; i++) this.layer0[i].activation = input[i];
    for (var i = 0; i < this.layers.length; i++) this.layers[i].updateActivation();
  }

  Evaluate(input, target_output, alpha){
    this.predict(input);
    return this.backpropagate(target_output, alpha);
  }

  backpropagate(target_output, alpha){
    var cost = this.layers[this.layers.length - 1].calculateOutputError(target_output, alpha);
    for (var i = this.layers.length - 1; i >= 0; i--) this.layers[i].backpropagate();
    return cost;
  }

  output(){
    var output = [];
    for (var i = 0; i < this.layers[this.layers.length - 1].neurons.length; i++) output[i] = this.layers[this.layers.length - 1].neurons[i].activation;
    return output;
  }

  learnFromYourMistakes(){
    for (var i = 0; i < this.layers.length; i++) this.layers[i].learnFromYourMistakes();
  }

  numParameters(){
    var sum = 0;
    for (var i = 0; i < this.layers.length; i++) sum += this.layers[i].numParameters();
    return sum;
  }

  static weightToColor(w){
    w = Math.sigmoid(w);
    if (w > 0.5){
      return "rgb(0," + (370*w - 115) + ",0)"
    } else {
      return "rgb(" + (255 - 370*w) + ",0,0)"
    }
  }

  draw(context, x, y, w, h){
    context.beginPath();
    context.fillStyle = "black";
    context.strokeStyle = "black";
    context.rect(x, y, w, h);
    context.fill();
    context.stroke();
    context.closePath();
    context.lineWidth = 1;
    context.lineJoin = "round";
    const num_layers = this.layers.length + 1;
    var layer_x_j, layer_x_i, step_y_j, step_y_i;
    step_y_j = h / (this.layer0.length + 1);
    for (var l = 0; l < this.layers.length; l++){
      layer_x_j = (l + 0.5) * w / num_layers + x;
      layer_x_i = (l + 1.5) * w / num_layers + x;
      step_y_i = h / (this.layers[l].neurons.length + 1);
      for (var i = 0; i < this.layers[l].neurons.length; i++){
        for (var j = 0; j < this.layers[l].neurons[i].children.length; j++){
          context.beginPath();
          context.strokeStyle = NeuronModel.weightToColor(this.layers[l].neurons[i].weights[j]);
          context.moveTo(layer_x_j, y + (j + 1) * step_y_j);
          context.lineTo(layer_x_i, y + (i + 1) * step_y_i);
          context.closePath();
          context.stroke();
        }
      }
      step_y_j = step_y_i;
    }
    context.fillStyle = "white";
    context.strokeStyle = "white";
    for (var l = 0; l < this.layers.length; l++){
      layer_x_i = (l + 1.5) * w / num_layers + x;
      step_y_i = h / (this.layers[l].neurons.length + 1);
      for (var i = 0; i < this.layers[l].neurons.length; i++){
        context.beginPath();
        var gray = this.layers[l].neurons[i].activation * 255;
        context.fillStyle = "rgb("+gray+","+gray+","+gray+")";
        context.arc(layer_x_i, y + (i + 1) * step_y_i, 0.4 * Math.min(w / num_layers, step_y_i), 0, 2 * Math.PI);
        context.stroke();
        context.fill();
      }
    }
    layer_x_i = 0.5 * w / num_layers + x;
    step_y_i = h / (this.layer0.length + 1);
    for (var i = 0; i < this.layer0.length; i++){
      context.beginPath();
      var gray = this.layer0[i].activation * 255;
      context.fillStyle = "rgb("+gray+","+gray+","+gray+")";
      context.arc(layer_x_i, y + (i + 1) * step_y_i, 0.4 * Math.min(w / num_layers, step_y_i), 0, 2 * Math.PI);
      context.stroke();
      context.fill();
    }
  }

  /*
  <neuronmodel> := <int> <int> [ <layer> ]
  */
  writeText(){
    for (var i = 0; i < this.layer0.length; i++) this.layer0[i].value = i;
    var str = intToString(this.layer0.length) + intToString(this.layers.length);
    var index = new Int(this.layer0.length);
    for (var i = 0; i < this.layers.length; i++) str += this.layers[i].writeText(index);
    return str;
  }

  static readText(str){
    var output = new NeuronModel(stringToInt(str.substr(0, 4)));
    var dict = [];
    for (var i = 0; i < output.layer0.length; i++) dict[i] = output.layer0[i];
    var num_layers = stringToInt(str.substr(4, 4));
    var index = new Int(8);
    for (var l = 0; l < num_layers; l++) output.layers[l] = Layer.readText(str, dict, index);
    return output;
  }
}
class Layer {
  constructor(dim, activation_function){
    this.neurons = [];
    for (var i = 0; i < dim; i++) this.neurons[i] = new Neuron();
    this.activation_function = activation_function.toLowerCase();
    this.sigmoid = Math[this.activation_function];
    this.diff_sigmoid = Math["diff_" + this.activation_function];
  }

  connectFull(other){
    for (var i = 0; i < this.neurons.length; i++){
      for (var j = 0; j < other.neurons.length; j++) this.neurons[i].addChild(other.neurons[j]);
    }
  }

  updateActivation(){
    for (var i = 0; i < this.neurons.length; i++) this.neurons[i].updateActivation(this.sigmoid);
  }

  calculateOutputError(target_output, alpha){
    var cost = 0;
    for (var i = 0; i < this.neurons.length; i++){
      this.neurons[i].error_potential = alpha * (this.neurons[i].activation - target_output[i]);
      cost += Math.sq(this.neurons[i].activation - target_output[i]);
    }
    return cost;
  }

  backpropagate(){
    for (var i = 0; i < this.neurons.length; i++) this.neurons[i].backpropagate(this.diff_sigmoid);
  }

  learnFromYourMistakes(){
    for (var i = 0; i < this.neurons.length; i++) this.neurons[i].learnFromYourMistakes();
  }

  numParameters(){
    var sum = 0;
    for (var i = 0; i < this.neurons.length; i++) sum += this.neurons[i].numParameters();
    return sum;
  }

  /*
    <layer> := <int> <string> <int> [ <neuron> ]
  */
  writeText(index){
    var str = intToString(this.activation_function.length) + this.activation_function + intToString(this.neurons.length);
    for (var i = 0; i < this.neurons.length; i++) str += this.neurons[i].writeText(index);
    return str;
  }

  static readText(str, dict, index){
    var len = stringToInt(str.substr(index.i, 4));
    var activation_function = str.substr(index.i + 4, len);
    index.i += 4 + len;
    var dim = stringToInt(str.substr(index.i, 4));
    index.i += 4;
    var output = new Layer(dim, activation_function);
    for (var i = 0; i < dim; i++) output.neurons[i] = Neuron.readText(str, dict, index);
    return output;
  }
}
class Neuron {
  constructor(){
    this.children = [];
    this.weights = [];
    this.bias = Math.random() - 0.5;
    this.potential = 0;
    this.activation = 0;

    this.error_weights = [];
    this.error_bias = 0;
    this.error_potential = 0;
  }

  addChild(other){
    this.children.push(other);
    this.weights.push(Math.random() - 0.5);
    this.error_weights.push(0);
  }

  updateActivation(activation_function){
    this.potential = this.bias;
    for (var i = 0; i < this.children.length; i++) this.potential += this.weights[i] * this.children[i].activation;
    this.activation = activation_function(this.potential);
  }

  backpropagate(f){
    this.error_potential *= f(this.potential);
    for (var i = 0; i < this.children.length; i++){
      this.children[i].error_potential += this.error_potential * this.weights[i];
      this.error_weights[i] += this.error_potential * this.children[i].activation;
    }
    this.error_bias += this.error_potential;
    this.error_potential = 0;
  }

  learnFromYourMistakes(){
    this.bias += this.error_bias;
    this.error_bias = 0;
    for (var i = 0; i < this.weights.length; i++){
      this.weights[i] += this.error_weights[i];
      this.error_weights[i] = 0;
    }
  }

  numParameters(){
    return this.weights.length + 1;
  }

  /*
    <neuron> := <float> <int> [ <int> <float> ]
  */
  writeText(index){
    this.value = (index.i++);
    var str = floatToString(this.bias) + intToString(this.children.length);
    for (var i = 0; i < this.children.length; i++) str += intToString(this.children[i].value) + floatToString(this.weights[i]);
    return str;
  }

  static readText(str, dict, index){
    var output = new Neuron();
    dict.push(output);
    output.bias = stringToFloat(str.substr(index.i, 4));
    var num = stringToInt(str.substr(index.i + 4, 4));
    index.i += 8;
    for (var i = 0; i < num; i++){
      output.children[i] = dict[stringToInt(str.substr(index.i, 4))];
      output.weights[i] = stringToFloat(str.substr(index.i + 4, 4));
      output.error_weights[i] = 0;
      index.i += 8;
    }
    return output;
  }
}
class InputNeuron {
  constructor(){
    this.activation = 0;
    this.error_potential = 0;
  }
}
class Int {
  constructor(i = 0){
    this.i = i;
  }
}
class Dataplotter {
  constructor(x = 20, y = 20, wid = 400, hig = 200, color = "red"){
    this.x = new Vector2(x, y);
    this.pxlW = wid - 20;
    this.pxlH = hig - 20;
    this.color = color;
    this.data = [];
    this.Xstep = 1;
    this.cycle = 0;
    this.Xmin = 0;
    this.Xmax = 0;
    this.Ymin = 0;
    this.Ymax = 0.00000000001;
    this.lastDataPoint = 0;
  }

  Push(y){
    this.lastDataPoint = y;
    if (y > this.Ymax) this.Ymax = y;
    if (y < this.Ymin) this.Ymin = y;
    if ((++this.cycle) != this.Xstep) return;
    this.cycle = 0;
    this.data.push(y);
    if (this.data.length == 512){
      var temp = [];
      for (var i = 0; i < this.data.length; i += 2) temp.push(this.data[i]);
      this.data = temp.slice();
      this.Xstep *= 2;
    }
    this.Xmax += this.Xstep;
  }

  draw(context){
    context.lineWidth = 1;
    context.beginPath();
    context.fillStyle = "white";
    context.strokeStyle = "black";
    context.rect(this.x.x, this.x.y, this.pxlW + 20, this.pxlH + 20);
    context.fill();
    context.stroke();
    context.closePath();
    context.beginPath();
    context.fillStyle = "lightgray";
    context.rect(this.x.x + 10, this.x.y + 10, this.pxlW, this.pxlH);
    context.fill();
    context.closePath();
    context.beginPath();
    context.strokeStyle = this.color;
    context.moveTo(-this.Xmin / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, (this.data[0] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    for (var i = 1; i < this.data.length; i++) context.lineTo((i * this.Xstep - this.Xmin) / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, (this.data[i] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    context.lineTo((i * this.Xstep + this.cycle - this.Xmin) / (this.Xmax - this.Xmin) * this.pxlW + this.x.x + 10, (this.lastDataPoint - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    context.moveTo(0, 0);
    context.closePath();
    context.stroke();
    context.fillStyle = "black";
    context.textAlign = "center";
    context.font = "10px Arial";
    context.fillText(this.Ymax, this.x.x + this.pxlW * 0.5, this.x.y + 9);
    context.fillText(this.Ymin, this.x.x + this.pxlW * 0.5, this.x.y + this.pxlH + 18);
  }

  Clear(){
    this.data = [];
    this.Xstep = 1;
    this.cycle = 0;
    this.Xmin = 0;
    this.Xmax = 0;
    this.Ymin = 0;
    this.Ymax = 0.00000000001;
    this.lastDataPoint = 0;
  }
}
class ContinuesDataPlotter {
  constructor(dim, x = 20, y = 20, wid = 400, hig = 200, color = "red"){
    this.x = new Vector2(x, y);
    this.pxlW = wid - 20;
    this.pxlH = hig - 20;
    this.Ymin = -0.00001;
    this.Ymax = 0.00001;
    this.max_dim = dim;
    this.data = [];
    this.Xstep = this.pxlW / (dim - 1);
    this.color = color;
  }

  Push(y){
    this.data.push(y);
    if (y > this.Ymax) this.Ymax = y;
    if (y < this.Ymin) this.Ymin = y;
    if (this.data.length == this.max_dim) this.data.shift();
  }

  draw(ctx){
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.fillStyle = "white";
    ctx.strokeStyle = "black";
    ctx.rect(this.x.x, this.x.y, this.pxlW + 20, this.pxlH + 20);
    ctx.fill();
    ctx.stroke();
    ctx.closePath();
    ctx.beginPath();
    ctx.fillStyle = "lightgray";
    ctx.rect(this.x.x + 10, this.x.y + 10, this.pxlW, this.pxlH);
    ctx.fill();
    ctx.closePath();
    ctx.beginPath();
    ctx.strokeStyle = this.color;
    ctx.moveTo(this.x.x + 10, (this.data[0] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    for (var i = 1; i < this.data.length; i++) ctx.lineTo(i * this.Xstep + this.x.x + 10, (this.data[i] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
    ctx.moveTo(0, 0);
    ctx.closePath();
    ctx.stroke();
    ctx.fillStyle = "black";
    ctx.textAlign = "center";
    ctx.font = "10px Arial";
    ctx.fillText(this.Ymax, this.x.x + this.pxlW * 0.5, this.x.y + 9);
    ctx.fillText(this.Ymin, this.x.x + this.pxlW * 0.5, this.x.y + this.pxlH + 18);
  }

  Clear(){
    this.data = [];
    this.Ymin = -0.00001;
    this.Ymax = 0.00001;
  }
}
class MultyContinuesDataPlotter {
  constructor(num_curves, dim, x = 20, y = 20, wid = 400, hig = 200, color = ["red"]){
    this.x = new Vector2(x, y);
    this.pxlW = wid - 20;
    this.pxlH = hig - 20;
    this.Ymin = -0.00001;
    this.Ymax = 0.00001;
    this.max_dim = dim;
    this.data = [];
    this.num_curves = num_curves;
    this.Xstep = this.pxlW / (dim - 1);
    this.color = color;
    while (this.color.length < num_curves) this.color.push("red");
  }

  Push(y){
    if (y.length != this.num_curves) return;
    this.data.push(y);
    for (var i = 0; i < this.num_curves; i++){
      if (y[i] > this.Ymax) this.Ymax = y[i];
      if (y[i] < this.Ymin) this.Ymin = y[i];
    }
    if (this.data.length == this.max_dim) this.data.shift();
  }

  draw(ctx){
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.fillStyle = "white";
    ctx.strokeStyle = "black";
    ctx.rect(this.x.x, this.x.y, this.pxlW + 20, this.pxlH + 20);
    ctx.fill();
    ctx.stroke();
    ctx.closePath();
    ctx.beginPath();
    ctx.fillStyle = "lightgray";
    ctx.rect(this.x.x + 10, this.x.y + 10, this.pxlW, this.pxlH);
    ctx.fill();
    ctx.closePath();
    for (var d = 0; d < this.num_curves; d++){
      ctx.beginPath();
      ctx.strokeStyle = this.color[d];
      ctx.moveTo(this.x.x + 10, (this.data[0][d] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      for (var i = 1; i < this.data.length; i++) ctx.lineTo(i * this.Xstep + this.x.x + 10, (this.data[i][d] - this.Ymax) / (this.Ymin - this.Ymax) * this.pxlH + this.x.y + 10);
      ctx.moveTo(0, 0);
      ctx.closePath();
      ctx.stroke();
    }
    ctx.fillStyle = "black";
    ctx.textAlign = "center";
    ctx.font = "10px Arial";
    ctx.fillText(this.Ymax, this.x.x + this.pxlW * 0.5, this.x.y + 9);
    ctx.fillText(this.Ymin, this.x.x + this.pxlW * 0.5, this.x.y + this.pxlH + 18);
  }

  Clear(){
    this.data = [];
    this.Ymin = -0.00001;
    this.Ymax = 0.00001;
  }
}
class Vector2 {
  constructor(x, y){
    this.x = x;
    this.y = y;
  }
  Multpl(x){
    return new Vector2(this.x * x, this.y * x);
  }
  Add(v){
    return new Vector2(this.x + v.x, this.y + v.y);
  }
  Sub(v){
    return new Vector2(this.x - v.x, this.y - v.y);
  }
  Dist(v){
    return Math.hypot(this.x - v.x, this.y - v.y);
  }
  Dot(v){
    return this.x * v.x + this.y * v.y;
  }
  Crs(v){
    return this.x * v.y - this.y * v.x;
  }
  rotate(phi){
    return new Vector2(this.x * Math.cos(phi) - this.y * Math.sin(phi), this.x * Math.sin(phi) + this.y * Math.cos(phi));
  }
  static Angle(phi){
    return new Vector2(Math.cos(phi), Math.sin(phi));
  }
  get magnitude(){
    return Math.hypot(this.x, this.y);
  }
  get normalized(){
    return this.Multpl(1/this.magnitude);
  }
  static get zero(){
    return new Vector2(0, 0);
  }
}
