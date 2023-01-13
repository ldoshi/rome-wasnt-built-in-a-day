let _COLOR_MAP = {
    0: "#FFFFFF",
    1: "#000000",
    2: "#550000",
}

let _TOP_ROW_HALF_BLOCK_COLOR = "#007700";
let _TOP_ROW_TRIANGLE_COLOR = "#DD0000";

let _AXIS_LABEL_BACKGROUND_COLOR = "#CCCCCC"
let _AXIS_LABEL_COLOR = "#000000"
let _AXIS_FONT = "bold 32px Arial";

let _BLOCK_BORDER_WIDTH = 1;

function render_array_2d(data, canvas_id, top_row_half_block_positions=null, top_row_triangle_positions=null) {
    let canvas = $(`#${canvas_id}`)[0]
    let width = canvas.width;
    // Add a row to the height and width for the axes labels.
    let block_size = width/(data[0].length + 1);
    let has_special_top_row = (top_row_half_block_positions != null || top_row_triangle_positions != null );
    height = block_size * (data.length + 1);
    if (has_special_top_row) {
	height += block_size
    }
    canvas.height = height;

    let ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0); 
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, width, height);

    // Label axes.
    for (let y = 0; y < data.length; y++) {
	ctx.fillStyle = _AXIS_LABEL_BACKGROUND_COLOR;
	ctx.fillRect(_BLOCK_BORDER_WIDTH, y * block_size + _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH);
	ctx.fillStyle = _AXIS_LABEL_COLOR;
	ctx.font = _AXIS_FONT;
	ctx.fillText(y, block_size/4, (y + 3/4) * block_size);
    }
    for (let x = 0; x < data[0].length; x++) {
	ctx.fillStyle = _AXIS_LABEL_BACKGROUND_COLOR;
	ctx.fillRect((x + 1) * block_size + _BLOCK_BORDER_WIDTH, data.length * block_size + _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH);
	ctx.fillStyle = _AXIS_LABEL_COLOR;
	ctx.font = _AXIS_FONT;
	ctx.fillText(x, (x + 1) * block_size + block_size/4, (data.length + 3/4) * block_size);
    }
       
    if (top_row_half_block_positions != null) {
	for (let i = 0; i < top_row_half_block_positions.length; i++) {
	    // Add one since the left-most column is the axes label.
	    let position = top_row_half_block_positions[i] + 1;
	    ctx.fillStyle = _TOP_ROW_HALF_BLOCK_COLOR;
	    ctx.fillRect(position * block_size + _BLOCK_BORDER_WIDTH, block_size/2 + _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH);
	}
    }

    if (top_row_triangle_positions != null) {
	for (let i = 0; i < top_row_triangle_positions.length; i++) {
	    // Add one since the left-most column is the axes label.
	    let position = top_row_triangle_positions[i] + 1;
	    ctx.fillStyle = _TOP_ROW_TRIANGLE_COLOR;
	    ctx.beginPath();
	    ctx.moveTo(position * block_size + _BLOCK_BORDER_WIDTH, _BLOCK_BORDER_WIDTH);
	    ctx.lineTo((position + 1) * block_size - _BLOCK_BORDER_WIDTH, _BLOCK_BORDER_WIDTH);
	    ctx.lineTo((position + .5) * block_size, block_size / 2 - _BLOCK_BORDER_WIDTH);
	    ctx.fill();
	}
    }
    
    let y_shift = has_special_top_row ? 1 : 0;
    for (let y = 0; y < data.length; y++) {
	// Add one to x since the left-most column is the axes label.
	for (let x = 0; x < data[y].length; x++) {
	    ctx.fillStyle = _COLOR_MAP[data[y][x]];
	    ctx.fillRect((x + 1) * block_size + _BLOCK_BORDER_WIDTH, (y + y_shift) * block_size + _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH, block_size - 2 * _BLOCK_BORDER_WIDTH);
	}
    }

}
