document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');
        });
    });

    // Add Memory
    const addBtn = document.getElementById('addMemoryBtn');
    const memoryInput = document.getElementById('memoryInput');
    const msgBox = document.getElementById('addMessage');

    addBtn.addEventListener('click', async () => {
        const text = memoryInput.value.trim();
        if (!text) return;

        setLoading(addBtn, true);
        msgBox.className = 'message hidden';

        try {
            const res = await fetch('/api/episodes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ episode_body: text })
            });
            
            if (res.ok) {
                msgBox.textContent = 'Memory flawlessly injected into the neural graph.';
                msgBox.className = 'message success';
                memoryInput.value = '';
            } else {
                throw new Error('Server returned an error');
            }
        } catch (error) {
            msgBox.textContent = 'Failed to index memory. Is the server running?';
            msgBox.className = 'message error';
            console.error(error);
        } finally {
            setLoading(addBtn, false);
        }
    });

    // Search Graph
    const searchBtn = document.getElementById('searchBtn');
    const searchInput = document.getElementById('searchInput');
    const resultsBox = document.getElementById('searchResults');

    const doSearch = async () => {
        const query = searchInput.value.trim();
        if (!query) return;

        resultsBox.innerHTML = '<div class="loader" style="margin: 2rem auto;"></div>';
        resultsBox.classList.remove('empty');

        try {
            const res = await fetch(`/api/search?query=${encodeURIComponent(query)}`);
            if (!res.ok) throw new Error('Search failed');
            const data = await res.json();
            
            renderResults(data);
        } catch (error) {
            resultsBox.innerHTML = `<p class="placeholder-text error" style="color:var(--error)">Search failed. Disconnected from core.</p>`;
            console.error(error);
        }
    };

    searchBtn.addEventListener('click', doSearch);
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doSearch();
    });

    function renderResults(data) {
        resultsBox.innerHTML = '';
        const { nodes, edges } = data;

        if (nodes.length === 0 && edges.length === 0) {
            resultsBox.innerHTML = '<p class="placeholder-text">No resonant memories found.</p>';
            return;
        }

        // Render Nodes
        nodes.forEach(node => {
            const el = document.createElement('div');
            el.className = 'result-card';
            el.innerHTML = `
                <div class="card-title">
                    <span class="badge">Node</span>
                    ${node.name}
                </div>
                <div class="card-body">${node.summary || '<i>No summary stored</i>'}</div>
            `;
            resultsBox.appendChild(el);
        });

        // Render Edges
        edges.forEach(edge => {
            const el = document.createElement('div');
            el.className = 'result-card';
            el.style.borderLeft = '3px solid var(--accent)';
            el.innerHTML = `
                <div class="card-title" style="color:#a79ff7;">
                    <span class="badge" style="background:rgba(255,255,255,0.1); color:white;">Relation</span>
                    Fact
                </div>
                <div class="card-body">"${edge.fact}"</div>
            `;
            resultsBox.appendChild(el);
        });
    }

    function setLoading(btnElement, isLoading) {
        const text = btnElement.querySelector('.btn-text');
        const loader = btnElement.querySelector('.loader');
        if (isLoading) {
            text.classList.add('hidden');
            loader.classList.remove('hidden');
        } else {
            text.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    }
});
