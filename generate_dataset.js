class Wave {
  constructor(x, angular_frequency){
    this.x = x;
    this.angular_frequency = angular_frequency;
    this.waveLength = 2 * Math.PI / angular_frequency;
  }

  next(){
    return 0.5 * Math.sin((this.x++) * this.angular_frequency);
  }
}

function exportDataFile(waveLengths = [20, 30, 40, 50], pointsPerWave = 60, wavesPerWaveLength, fileName = "training"){
  var training_data_text = "";
  var training_labels_text = "";
  var training_waves = [];
  for (var i = 0; i < waveLengths.length; i++){
    for (var j = 0; j < wavesPerWaveLength; j++){
      training_data_text += waveLengths[i] + ";";
      training_waves.push(new Wave(Math.random() * waveLengths[i], 2 * Math.PI / waveLengths[i]));
      training_labels_text += training_waves[training_waves.length - 1].x + ";";
    }
  }
  training_data_text = training_data_text.slice(0, -1) + "\n" + training_labels_text.slice(0, -1) + "\n";
  for (var i = 0; i < pointsPerWave; i++){
    for (var j = 0; j < training_waves.length; j++){
      training_data_text += (training_waves[j].next() + 0.4 * (Math.random() - 0.5)) + ";";
    }
      training_data_text = training_data_text.slice(0, -1) + "\n";
  }
  exportAsCSV(training_data_text, fileName + "_data");
}
exportDataFile([20, 30, 40, 50], 60, 2000, "training");
exportDataFile([20, 30, 40, 50], 60, 500, "test");

function exportAsCSV(text, filename = "test"){
  let textFileAsBlob = new Blob([text], { type: "text/plain" });
  let downloadLink = document.createElement('a');
  downloadLink.download = filename + ".csv";
  downloadLink.innerHTML = 'Download File';

  if (window.webkitURL != null) {
    downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
  } else {
    downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
    downloadLink.style.display = 'none';
    document.body.appendChild(downloadLink);
  }
  downloadLink.click();
}
