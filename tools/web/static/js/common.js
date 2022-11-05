// TODO(lyric): Add optional arguments for additional things to draw
// like for action inversion analysis.

let _COLOR_MAP = {
    0: "#FFFFFF",
    1: "#000000",
    2: "#550000",
}

let _BLOCK_BORDER_WIDTH = 1;

function render_array_2d(data, canvas_id) {
    let canvas = $(`#${canvas_id}`)[0]
    let width = canvas.width;
    let block_size = width/data[0].length;
    height = block_size * data.length;
    canvas.height = block_size * data.length;

    let ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0); 
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, width, height);
    

    for (let y = 0; y < data.length; y++) {
	for (let x = 0; x < data[y].length; x++) {
	    ctx.fillStyle = _COLOR_MAP[data[y][x]];
	    ctx.fillRect(x * block_size + _BLOCK_BORDER_WIDTH, y * block_size + _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH);
	}
    }
}
