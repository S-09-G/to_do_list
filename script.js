document.addEventListener('DOMContentLoaded', () => {
    const newTask = document.getElementById('new-task');
    const addTask = document.getElementById('add-task');
    const taskList = document.getElementById('task-list');
    const filterActive = document.getElementById('filter-active');
    const filterCompleted = document.getElementById('filter-completed');
    const filterAll = document.getElementById('filter-all');

    const tasks = JSON.parse(localStorage.getItem('tasks')) || [];

    function renderTasks() {
        taskList.innerHTML = '';
        tasks.forEach((task, index) => {
            const li = document.createElement('li');
            li.textContent = task.text;
            li.id = `task-${index}`;
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = task.completed;
            checkbox.addEventListener('change', () => {
                task.completed = checkbox.checked;
                localStorage.setItem('tasks', JSON.stringify(tasks));
                renderTasks();
            });
            li.appendChild(checkbox);
            const deleteButton = document.createElement('button');
            deleteButton.textContent = 'Delete';
            deleteButton.addEventListener('click', () => {
                tasks.splice(index, 1);
                localStorage.setItem('tasks', JSON.stringify(tasks));
                renderTasks();
            });
            li.appendChild(deleteButton);
            taskList.appendChild(li);
        });
    }

    addTask.addEventListener('click', () => {
        const taskText = newTask.value.trim();
        if (taskText) {
            tasks.push({ text: taskText, completed: false });
            newTask.value = '';
            localStorage.setItem('tasks', JSON.stringify(tasks));
            renderTasks();
        }
    });

    filterActive.addEventListener('click', () => {
        tasks.forEach(task => {
            const li = document.getElementById(`task-${tasks.indexOf(task)}`);
            if (li) {
                li.style.display = task.completed ? 'none' : '';
            }
        });
    });

    filterCompleted.addEventListener('click', () => {
        tasks.forEach(task => {
            const li = document.getElementById(`task-${tasks.indexOf(task)}`);
            if (li) {
                li.style.display = task.completed ? '' : 'none';
            }
        });
    });

    filterAll.addEventListener('click', () => {
        tasks.forEach(task => {
            const li = document.getElementById(`task-${tasks.indexOf(task)}`);
            if (li) {
                li.style.display = '';
            }
        });
    });

    renderTasks();
});
