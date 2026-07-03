document.addEventListener('DOMContentLoaded', function() {
    
    // Autocomplete Logic for Players
    const playerInput = document.getElementById('playerSearchInput');
    const playerResults = document.getElementById('playerAutocomplete');
    
    if (playerInput && playerResults) {
        playerInput.addEventListener('input', function() {
            const query = this.value;
            if (query.length < 2) {
                playerResults.style.display = 'none';
                return;
            }
            
            fetch(`/api/players/search?q=${encodeURIComponent(query)}`)
                .then(res => res.json())
                .then(data => {
                    playerResults.innerHTML = '';
                    if (data.length > 0) {
                        data.forEach(item => {
                            const div = document.createElement('div');
                            div.className = 'autocomplete-item';
                            div.textContent = item;
                            div.onclick = function() {
                                window.location.href = `/player/${encodeURIComponent(item)}`;
                            };
                            playerResults.appendChild(div);
                        });
                        playerResults.style.display = 'block';
                    } else {
                        playerResults.style.display = 'none';
                    }
                });
        });
        
        // Hide autocomplete when clicking outside
        document.addEventListener('click', function(e) {
            if (e.target !== playerInput && e.target !== playerResults) {
                playerResults.style.display = 'none';
            }
        });
    }
    
    // Autocomplete Logic for Teams
    const teamInput = document.getElementById('teamSearchInput');
    const teamResults = document.getElementById('teamAutocomplete');
    
    if (teamInput && teamResults) {
        teamInput.addEventListener('input', function() {
            const query = this.value;
            if (query.length < 2) {
                teamResults.style.display = 'none';
                return;
            }
            
            fetch(`/api/teams/search?q=${encodeURIComponent(query)}`)
                .then(res => res.json())
                .then(data => {
                    teamResults.innerHTML = '';
                    if (data.length > 0) {
                        data.forEach(item => {
                            const div = document.createElement('div');
                            div.className = 'autocomplete-item';
                            div.textContent = item;
                            div.onclick = function() {
                                window.location.href = `/team/${encodeURIComponent(item)}`;
                            };
                            teamResults.appendChild(div);
                        });
                        teamResults.style.display = 'block';
                    } else {
                        teamResults.style.display = 'none';
                    }
                });
        });
        
        document.addEventListener('click', function(e) {
            if (e.target !== teamInput && e.target !== teamResults) {
                teamResults.style.display = 'none';
            }
        });
    }
});
