// index.js
async function init() {
    const wasm = await WebAssembly.import('./main.wasm');
    const { DoThing } = wasm;

    const input = document.getElementById('myInput');
    const button = document.getElementById('myButton');
    const output = document.getElementById('myOutput');

    button.addEventListener('click', () => {
        const value = input.value;
        const result = DoThing(value);
        output.innerText = result;
    });
}

init();