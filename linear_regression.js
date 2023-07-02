class Matrix {
  constructor(height, width){
    this.m = [];
    for (var i = 0; i < height; i++){
      this.m[i] = [];
      for (var j = 0; j < width; j++) this.m[i][j] = 0;
    }
    this.width = width;
    this.height = height;
  }

  Prod(other){
    if (other.height != this.width) return null;
    var output = new Matrix(this.height, other.width);
    for (var i = 0; i < output.height; i++){
      for (var j = 0; j < output.width; j++){
        output.m[i][j] = 0;
        for (var k = 0; k < this.width; k++) output.m[i][j] += this.m[i][k] * other.m[k][j];
      }
    }
    return output;
  }

  Add(other){
    if (this.width != other.width || this.height != other.height) return;
    for (var i = 0; i < this.height; i++){
      for (var j = 0; j < this.width; j++) this.m[i][j] += other.m[i][j];
    }
  }

  Multpl(x){
    var output = new Matrix(this.height, this.width);
    for (var i = 0; i < this.height; i++){
      for (var j = 0; j < this.width; j++) output.m[i][j] = this.m[i][j] * x;
    }
    return output;
  }

  SwapRow(i, j){
    let temp = this.m[j];
    this.m[j] = this.m[i];
    this.m[i] = temp;
  }

  get transpose(){
    var output = new Matrix(this.width, this.height);
    for (var i = 0; i < this.height; i++){
      for (var j = 0; j < this.width; j++) output.m[j][i] = this.m[i][j];
    }
    return output;
  }

  get inverse(){
    if (this.width != this.height) return;
    var inverse = Matrix.identity(this.width);
    for (var x = 0; x < this.width; x++){
      var y;
      for (y = x; y < this.width && this.m[y][x] == 0; y++);
      if (y == this.width) return null;
      this.SwapRow(x, y);
      inverse.SwapRow(x, y);
      let factor = 1 / this.m[x][x];
      for (var i = 0; i < this.width; i++){
        this.m[x][i] *= factor;
        inverse.m[x][i] *= factor;
      }
      for (y = 0; y < this.width; y++){
        if (x == y) continue;
        factor = -this.m[y][x];
        for (var i = 0; i < this.width; i++){
          inverse.m[y][i] += factor * inverse.m[x][i];
          this.m[y][i] += factor * this.m[x][i];
        }
      }
    }
    return inverse;
  }

  static identity(dim){
    var output = new Matrix(dim, dim);
    for (var i = 0; i < dim; i++) output.m[i][i] = 1;
    return output;
  }

  static fromArray(arr){
    var output = new Matrix(arr.length, arr[0].length);
    for (var i = 0; i < output.height; i++){
      for (var j = 0; j < output.width; j++) output.m[i][j] = arr[i][j];
    }
    return output;
  }
}
class LinearModel {
  constructor(input_dim, output_dim){
    this.weight = new Matrix(output_dim, input_dim);
    this.bias = new Matrix(output_dim, 1);
    this.numerator = new Matrix(output_dim, input_dim);
    this.denominator = new Matrix(input_dim, input_dim);
    this.num_dataPoints = 0;
    this.totalX = new Matrix(input_dim, 1);
    this.totalY = new Matrix(output_dim, 1);
    this.input_dim = input_dim;
    this.output_dim = output_dim;
  }

  fit(){
    this.weight = this.numerator.Prod(this.denominator.inverse);
    this.bias = this.totalY.Multpl(1 / this.num_dataPoints);
    this.bias.Add(this.weight.Prod(this.totalX.Multpl(-1 / this.num_dataPoints)));
    this.numerator = new Matrix(this.output_dim, this.input_dim);
    this.denominator = new Matrix(this.input_dim, this.input_dim);
    this.num_dataPoints = 0;
    this.totalX = new Matrix(this.input_dim, 1);
    this.totalY = new Matrix(this.output_dim, 1);
  }

  Evaluate(input, target_output){
    let x = Matrix.fromArray([input]);
    let y = Matrix.fromArray([target_output]);
    this.numerator.Add(y.transpose.Prod(x));
    this.denominator.Add(x.transpose.Prod(x));
    this.totalX.Add(x.transpose);
    this.totalY.Add(y.transpose);
    this.num_dataPoints++;
  }

  predict(input){
    var output = this.weight.Prod(Matrix.fromArray([input]).transpose);
    output.Add(this.bias);
    return output.transpose.m[0];
  }
}
