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
  var training_data_text = "wave length;phase;";
  for (var i = 0; i < pointsPerWave; i++) training_data_text += i + ((i == pointsPerWave - 1) ? "\n" : ";");
  var training_waves = [];
  for (var i = 0; i < waveLengths.length; i++){
    for (var j = 0; j < wavesPerWaveLength; j++){
      let wave = new Wave(Math.random() * waveLengths[i], 2 * Math.PI / waveLengths[i]);
      training_data_text += waveLengths[i] + ";" + wave.x + ";";
      for (var k = 0; k < pointsPerWave; k++){
        training_data_text += (wave.next() + 0.4 * (Math.random() - 0.5)) + ((k == pointsPerWave - 1) ? "\n" : ";");
      }
    }
  }
  exportAsCSV(training_data_text, fileName + "_data_transposed");
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
