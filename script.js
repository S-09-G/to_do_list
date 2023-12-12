const newTaskInput = document.getElementById('new-task');
const addTaskButton = document.getElementById('add-task');
const taskList = document.getElementById('task-list');

addTaskButton.addEventListener('click', () => {
    const newTask = newTaskInput.value;
    if (newTask) {
        // Create a new list item
        const listItem = document.createElement('li');
        listItem.textContent = newTask;

        // Add a button to complete the task
        const completeButton = document.createElement('button');
        completeButton.textContent = 'Complete';
        completeButton.addEventListener('click', () => {
            listItem.classList.add('completed');
        });

        // Add a button to remove the task
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        removeButton.addEventListener('click', () => {
            listItem.remove();
        });

        // Append buttons and list item to the list
        listItem.appendChild(completeButton);
        listItem.appendChild(removeButton);
        taskList.appendChild(listItem);

        // Clear the input field
        newTaskInput.value = '';
    }
});
