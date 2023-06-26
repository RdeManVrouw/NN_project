Math.argmax = function (array){
  if (array.length == 0) return undefined;
  var index = 0, max = array[0];
  for (var i = 1; i < array.length; i++){
    if (array[i] > max){
      index = i;
      max = array[i];
    }
  }
  return index;
}
Math.sq = function (x){
  return x * x;
}

Math.sigmoid = function (x){
  return 1.0 / (1 + Math.exp(-x));
}
Math.relu = function (x){
  return (x < 0) ? 0.0 : x;
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
Array.prototype.size = function (){
  var sum = 0;
  for (var i = 0; i < this.length; i++){
    if ((typeof this[i]) == "number"){
      sum++;
    } else {
      sum += this[i].size();
    }
  }
  return sum;
}

class Neuralnet {
  constructor(numInputs){
    this.dim = [numInputs];
    this.sigmoids = [];
    this.diff_sigmoids = [];
    this.a0 = [];
  }

  Evaluate(input){
    this.a0 = input;
    for (var l = 0; l < this.dim.length - 1; l++){
      for (var i = 0; i < this.dim[l + 1]; i++){
        this.z[l][i] = this.b[l][i];
        for (var j = 0; j < this.dim[l]; j++) this.z[l][i] += input[j] * this.w[l][i][j];
        this.a[l][i] = this.sigmoids[l](this.z[l][i]);
      }
      input = this.a[l];
    }
    return input;
  }

  mutate(numW = 4, numB = 3, error = 0.1){
    for (var num = 0; num < numW; num++){
      var l = Math.floor(Math.random() * (this.dim.length - 1)) + 1;
      var i = Math.floor(Math.random() * this.dim[l]);
      var j = Math.floor(Math.random() * this.dim[l - 1]);
      this.w[l][i][j] += 2 * error * (Math.random() - 0.5);
    }
    for (var num = 0; num < numB; num++){
      var l = Math.floor(Math.random() * (this.dim.length - 1)) + 1;
      var i = Math.floor(Math.random() * this.dim[l]);
      this.b[l][i] += 2 * error * (Math.random() - 0.5);
    }
  }

  setup(){
    this.a = [];
    this.z = [];
    this.b = [];
    this.w = [];
    for (var l = 0; l < this.dim.length - 1; l++){
      this.z[l] = [];
      this.a[l] = [];
      this.b[l] = [];
      this.w[l] = [];
      for (var i = 0; i < this.dim[l + 1]; i++){
        this.z[l][i] = 0;
        this.a[l][i] = 0;
        this.b[l][i] = 0;
        this.w[l][i] = [];
        for (var j = 0; j < this.dim[l]; j++) this.w[l][i][j] = 0;
      }
    }
  }

  addLayer(l, activation_function = "sigmoid"){
    this.dim.push(l);
    this.sigmoids.push(Math[activation_function.toLowerCase()]);
    this.diff_sigmoids.push(Math["diff_" + activation_function.toLowerCase()]);
  }

  get copy(){
    var output = new Neuralnet(this.dim);
    for (var l = 1; l < this.dim.length; l++){
      for (var i = 0; i < this.dim[l]; i++){
        output.b[l][i] = this.b[l][i];
        for (var j = 0; j < this.dim[l - 1]; j++) output.w[l][i][j] = this.w[l][i][j];
      }
    }
    return output;
  }

  randomize(){
    for (var l = 0; l < this.dim.length - 1; l++){
      for (var i = 0; i < this.dim[l + 1]; i++){
        this.b[l][i] = 0.2 * (Math.random() - 0.5);
        for (var j = 0; j < this.dim[l]; j++) this.w[l][i][j] = 0.2 * (Math.random() - 0.5);
      }
    }
  }

  size(){
    return this.w.size() + this.b.size();
  }

  static weightToColor(w){
    w = Math.sigmoid(w);
    if (w > 0.5){
      return "rgb(0," + (370*w - 115) + ",0)"
    } else {
      return "rgb(" + (255 - 370*w) + ",0,0)"
    }
  }

  draw(x, y, w, h, color = false){
    ctx.beginPath();
    ctx.fillStyle = "black";
    ctx.strokeStyle = "black";
    ctx.rect(x, y, w, h);
    ctx.fill();
    ctx.stroke();
    ctx.closePath();
    ctx.lineWidth = 1;
    ctx.lineJoin = "round";
    var layer_x_j, layer_x_i, step_y_j, step_y_i;
    for (var l = 0; l < this.w.length; l++){
      layer_x_j = (l + 0.5) * w / this.dim.length + x;
      layer_x_i = (l + 1.5) * w / this.dim.length + x;
      step_y_j = h / (this.dim[l] + 1);
      step_y_i = h / (this.dim[l + 1] + 1);
      for (var i = 0; i < this.dim[l + 1]; i++){
        for (var j = 0; j < this.dim[l]; j++){
          ctx.beginPath();
          ctx.strokeStyle = Neuralnet.weightToColor(this.w[l][i][j]);
          ctx.moveTo(layer_x_j, y + (j + 1) * step_y_j);
          ctx.lineTo(layer_x_i, y + (i + 1) * step_y_i);
          ctx.closePath();
          ctx.stroke();
        }
      }
    }
    ctx.fillStyle = "white";
    ctx.strokeStyle = "white";
    var a = this.a0;
    for (var l = 0; l < this.dim.length; l++){
      layer_x_i = (l + 0.5) * w / this.dim.length + x;
      step_y_i = h / (this.dim[l] + 1);
      for (var i = 0; i < this.dim[l]; i++){
        ctx.beginPath();
        if (color) ctx.fillStyle = "rgb("+(a[i]*255)+","+(a[i]*255)+","+(a[i]*255)+")";
        ctx.arc(layer_x_i, y + (i + 1) * step_y_i, 0.4 * Math.min(w / this.dim.length, step_y_i), 0, 2 * Math.PI);
        ctx.stroke();
        ctx.fill();
      }
      a = this.a[l];
    }
  }
}
class Model {
  constructor(numInputs){
    this.dim = [numInputs];
    this.network = new Neuralnet(numInputs);
    this.error = new Neuralnet(numInputs);
  }

  addLayer(l, activation_function = "sigmoid", num_copies = 1){
    for (var i = 0; i < num_copies; i++){
      this.network.addLayer(l, activation_function);
      this.error.addLayer(l, activation_function);
      this.dim.push(l);
    }
  }

  setup(){
    this.network.setup();
    this.network.randomize();
    this.error.setup();
    this.l = this.dim.length - 1;
  }

  Evaluate(input, target_output, alpha = 0.1){
    var output = this.network.Evaluate(input);
    var cost = 0;
    for (var i = 0; i < this.dim[this.l]; i++){
      this.error.z[this.l - 1][i] = alpha * (target_output[i] - output[i]) * this.network.diff_sigmoids[this.l - 1](this.network.z[this.l - 1][i]);
      cost += Math.sq(output[i] - target_output[i]);
    }
    for (var l = this.l - 1; l > 0; l--){
      for (var j = 0; j < this.dim[l]; j++){
        this.error.z[l - 1][j] = 0;
        for (var i = 0; i < this.dim[l + 1]; i++) this.error.z[l - 1][j] += this.error.z[l][i] * this.network.w[l][i][j];
        this.error.z[l - 1][j] *= this.network.diff_sigmoids[l](this.network.z[l - 1][j]);
      }
    }
    var activation = this.network.a0;
    for (var l = 0; l < this.l; l++){
      for (var i = 0; i < this.dim[l + 1]; i++){
        this.error.b[l][i] += this.error.z[l][i];
        for (var j = 0; j < this.dim[l]; j++) this.error.w[l][i][j] += this.error.z[l][i] * activation[j];
      }
      activation = this.network.a[l];
    }
    return cost;
  }

  size(){
    return this.network.size();
  }

  get output(){
    return this.network.a[this.network.dim.length - 1];
  }

  learnFromYourMistakes(){
    for (var l = 0; l < this.dim.length - 1; l++){
      for (var i = 0; i < this.dim[l + 1]; i++){
        this.network.b[l][i] += this.error.b[l][i];
        this.error.b[l][i] = 0;
        for (var j = 0; j < this.dim[l]; j++){
          this.network.w[l][i][j] += this.error.w[l][i][j];
          this.error.w[l][i][j] = 0;
        }
      }
    }
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
  Dist(v){
    return Math.hypot(this.x - v.x, this.y - v.y);
  }
  rotate(phi){
    return new Vector2(this.x * Math.cos(phi) - this.y * Math.sin(phi), this.x * Math.sin(phi) + this.y * Math.cos(phi));
  }
  get magnitude(){
    return Math.sqrt(this.x*this.x + this.y*this.y);
  }
  get normalized(){
    return this.Multpl(1/this.magnitude);
  }
}

/*
var model = new Model(<int>)                  // number of input neurons
model.addLayer(<int>, [<string>, <int>])      // num neurons, activation function (default: sigmoid), number of repititions of this layer
// possible functions: "sigmoid", "relu", "leaky_relu", "elu", "tanh", "identity" (Not capital-sensitive)
model.setup()                                 // initializes the model

model.Evaluate(<array>, <array>, [<float>])   // input, target output, gradient factor alpha; updates error network. Used for training
model.learnFromYourMistakes()                 // updates network. Error network is reset
model.network.Evaluate(<array>)               // input; predicts the output best it can
model.network.mutate([<int>, <int>, <float>]) // number of weights effected, number of biases effected, step size

model.network.draw(<int>, <int>, <int>, <int>, [<boolean>]) // x, y, width, height, color

model.error                                   // accesses error network

var dataPlot = new Dataplotter(<int>, <int>, <int>, <int>, <string>)      // x, y, width, height, color
dataPlot.Push(<float>)                                                    // adds datapoint
dataPlot.draw()                                                           // duh

var dataPlot = new Dataplotter(<int>, <int>, <int>, <int>, <int>, <string>)      // window width, x, y, width, height, color
dataPlot.Push(<float>)                                                           // adds datapoint
dataPlot.draw()                                                                  // duh
*/
