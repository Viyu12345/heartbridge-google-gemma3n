$(document).ready(function() {
    // Start conversation button handler
    $('#start_conversation').on('click', function() {
        let person1_name = $('#person1_name').val();
        let person1_gender = $('#person1_gender').val();
        let person2_name = $('#person2_name').val();
        let person2_gender = $('#person2_gender').val();
        let topic = $('#topic').val();

        // Start the conversation on the server
        $.post('/start_conversation', {
            person1_name: person1_name,
            person1_gender: person1_gender,
            person2_name: person2_name,
            person2_gender: person2_gender,
            topic: topic
        }, function(data) {
            $('#conversation_output').text(data.message);
            // Begin polling for new audio files
            pollForAudio();
        });
    });

    // Stop conversation button handler
    $('#stop_conversation').on('click', function() {
        $.post('/stop_conversation', {}, function(data) {
            $('#conversation_output').text(data.message);
        });
    });

    // Polling function to check for new audio files from the server
    function pollForAudio() {
        setInterval(function() {
            $.getJSON('/get_next_audio', function(data) {
                console.log('Audio response:', data);  // Log the data received from the server
                if (data.audio_url) {
                    // Set the audio player source to the new audio file URL and play it
                    let audioPlayer = document.getElementById('audio_player');
                    audioPlayer.src = data.audio_url;
                    audioPlayer.play();
                    console.log(`Playing audio: ${data.audio_url}`);
                } else {
                    console.log('No new audio found');
                }
            });
        }, 7000);
    }

    // Input voice button handler
    $('#input_voice').on('click', function() {
        $.post('/input_voice', {}, function(data) {
            $('#topic').val(data.topic);
        });
    });
});
