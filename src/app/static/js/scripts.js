// src/app/static/js/scripts.js
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    var formData = new FormData();
    var resumes = document.getElementById('resumes').files;
    for (var i = 0; i < resumes.length; i++) {
        formData.append('resumes', resumes[i]);
    }

    var jobDescriptionFile = document.getElementById('job-description-file').files[0];
    if (jobDescriptionFile) {
        formData.append('job_description', jobDescriptionFile);
    } else {
        formData.append('job_description', document.getElementById('job-description-text').value);
    }

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        var resultDiv = document.getElementById('result');
        resultDiv.innerHTML = data;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
