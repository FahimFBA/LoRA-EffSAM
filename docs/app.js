(function () {
  // Mean test Dice and SD over 3 seeds (results/03_results.md, Table 1)
  var DATA = [
    { name: "Baseline", note: "256 px, ViT-T, plain decoder", single: [0.780, 0.006], tta: [0.791, 0.007] },
    { name: "+ 512 px", note: "full resolution", single: [0.800, 0.003], tta: [0.813, 0.005] },
    { name: "+ intensity augmentation", note: "", single: [0.805, 0.003], tta: [0.814, 0.005] },
    { name: "LoRA-EffSAM", note: "ViT-S, hybrid decoder, ViT skips, deep supervision", single: [0.819, 0.003], tta: [0.823, 0.003], final: true }
  ];
  var LO = 0.76, HI = 0.84;
  var plot = document.getElementById("ladder-plot");
  if (!plot) return;

  function pct(v) { return ((v - LO) / (HI - LO)) * 100; }

  var rows = DATA.map(function (d) {
    var row = document.createElement("div");
    row.className = "row" + (d.final ? " final" : "");
    row.innerHTML =
      '<div class="row-name"><b></b><span></span></div>' +
      '<div class="track"><div class="whisk"></div><div class="dot"></div></div>' +
      '<div class="row-val"></div>';
    row.querySelector("b").textContent = d.name;
    row.querySelector(".row-name span").textContent = d.note;
    plot.appendChild(row);
    return row;
  });

  var axis = document.createElement("div");
  axis.className = "axis";
  var ticks = "";
  for (var t = LO; t <= HI + 1e-9; t += 0.02) {
    ticks += '<i style="left:' + pct(t) + '%">' + t.toFixed(2) + "</i>";
  }
  axis.innerHTML = '<span></span><div class="axis-scale">' + ticks + "</div><span></span>";
  plot.appendChild(axis);

  function render(tta) {
    DATA.forEach(function (d, i) {
      var v = tta ? d.tta : d.single, m = v[0], sd = v[1], row = rows[i];
      row.querySelector(".dot").style.left = pct(m) + "%";
      var w = row.querySelector(".whisk");
      w.style.left = pct(m - sd) + "%";
      w.style.width = (pct(m + sd) - pct(m - sd)) + "%";
      row.querySelector(".row-val").innerHTML = m.toFixed(3) + "<span>± " + sd.toFixed(3) + "</span>";
    });
  }
  render(false);

  var btns = document.querySelectorAll(".ladder-ctl button");
  btns.forEach(function (b) {
    b.addEventListener("click", function () {
      btns.forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
      render(b.getAttribute("data-tta") === "1");
    });
  });

  document.querySelectorAll(".copy").forEach(function (b) {
    b.addEventListener("click", function () {
      var el = document.getElementById(b.getAttribute("data-copy"));
      var done = function () {
        b.textContent = "Copied";
        setTimeout(function () { b.textContent = "Copy"; }, 1600);
      };
      if (navigator.clipboard) navigator.clipboard.writeText(el.innerText).then(done, function () {});
    });
  });
})();
