% Simple utility to mark specific edge coordinates on left/right images.
% Usage:
%   match_debug_file('/path/to/left.png', '/path/to/right.png');
%   match_debug_file('/path/to/left.png', '/path/to/right.png', 'outputs');
function match_debug_file(leftImagePath, rightImagePath, outDir)
	if nargin < 3
		outDir = 'outputs';
	end
	if ~exist(outDir, 'dir')
		mkdir(outDir);
	end

	% Requested edge coordinates
	leftPt  = [451.348,	478.9];
	rightPt = [436.868, 478.907];

	% Load images
	leftImg  = imread(leftImagePath);
	rightImg = imread(rightImagePath);

	% Annotate left image
	f1 = figure('Visible', 'off');
	imshow(leftImg);
	hold on;
	plot(leftPt(1), leftPt(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
	text(leftPt(1)+5, leftPt(2)-10, sprintf('(%.2f, %.2f)', leftPt(1), leftPt(2)), ...
		'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
	title(sprintf('Left edge at (%.2f, %.2f)', leftPt(1), leftPt(2)));
	saveas(f1, fullfile(outDir, 'left_marked.png'));
	close(f1);

	% Annotate right image
	f2 = figure('Visible', 'off');
	imshow(rightImg);
	hold on;
	plot(rightPt(1), rightPt(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
	text(rightPt(1)+5, rightPt(2)-10, sprintf('(%.2f, %.2f)', rightPt(1), rightPt(2)), ...
		'Color', 'green', 'FontSize', 10, 'FontWeight', 'bold');
	title(sprintf('Right edge at (%.2f, %.2f)', rightPt(1), rightPt(2)));
	saveas(f2, fullfile(outDir, 'right_marked.png'));
	close(f2);

	fprintf('Saved annotated images to %s\n', outDir);
end
