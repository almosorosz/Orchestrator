// Execute the real sensor polling functions in a small DOM harness.
// No browser, internet connection or Pi hardware is required.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../brain/pi/fumehood_dashboard.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Check the complete dashboard script, not only the harnessed functions.
assert(!script.includes('demoSensors'));
assert(!script.includes('Math.random'));
const helpers = script.slice(script.indexOf('function effective('), script.indexOf('async function loadProbeConfig'));
const poll = script.slice(script.indexOf('async function pollSensors('), script.indexOf('setInterval(pollSensors'));
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, {textContent: '', className: '', classList: {toggle() {}}});
  return elements.get(id);
};
let response = [{id: 'probe-1', temp_c: 24.5, status: 'ok', type: 'temp'}, {id: 'door-gp5', state: 'closed', status: 'ok', type: 'door'}];
let mode = 'ok';
let reloadedCamera = 0;
const context = vm.createContext({
  console, Date, Set, AbortController, setTimeout, clearTimeout,
  document: {getElementById: element, querySelector: () => null, querySelectorAll: () => []},
  fetch: async (url, options) => {
    if (mode === 'fail') throw new Error('Cable unplugged');
    if (mode === 'hang') return new Promise((resolve, reject) => options.signal.addEventListener('abort', () => reject(new Error('Timeout'))));
    return {ok: mode !== 'http-error', json: async () => response};
  },
  loadProbeConfig() {}, loadHoodInfo() {}, loadCamera() { reloadedCamera++; },
  renderHistoryLog() {}, renderSensorTile() {}, updateSensorCards() {},
  readingVal: s => s.value ?? s.temp_c ?? null,
  isDoorSensor: s => s?.type === 'door',
  fmtVal: value => String(value), unitOf: () => '°C', classify: () => 'ok',
  fmtDuration: value => String(value),
});
vm.runInContext(`
  const CONFIG={graceReads:3};
  let sensors=[], connectionLost=false, lastSensorUpdate=null, sensorPollPending=false;
  let cameraAvailable=true, currentView='home', probeView=null, camOpen=false;
  const probeState={},history={};
  ${helpers}
  ${poll}
`, context);
const evaluate = code => vm.runInContext(code, context);

(async () => {
  await evaluate('pollSensors()');
  assert.equal(evaluate('effective("probe-1").value'), 24.5);
  assert.equal(evaluate('connectionLost'), false);
  mode = 'fail';
  await evaluate('pollSensors()');
  assert.equal(evaluate('connectionLost'), true);
  assert.equal(evaluate('effective("probe-1").offline'), true);
  assert.equal(evaluate('sensors.find(s=>s.type==="door").status'), 'offline');
  assert.equal(evaluate('sensors[0].temp_c'), null);
  assert.match(element('connectionBadge').textContent, /Sentinel offline.*Last update:/);
  mode = 'ok';
  await evaluate('pollSensors()');
  assert.equal(evaluate('effective("probe-1").value'), 24.5);
  assert.equal(element('connectionBadge').textContent, '');
  assert.equal(reloadedCamera, 1);
  response = [];
  await evaluate('pollSensors()');
  assert.equal(evaluate('effective("probe-1").offline'), true);
  assert.equal(evaluate('sensors[0].status'), 'offline');
  response = {not: 'a sensor array'};
  await evaluate('pollSensors()');
  assert.equal(evaluate('connectionLost'), true);
  mode = 'http-error';
  await evaluate('pollSensors()');
  assert.equal(evaluate('connectionLost'), true);
  mode = 'hang';
  const pending = evaluate('pollSensors()');
  await evaluate('pollSensors()'); // A second poll must not overlap the first.
  assert.equal(evaluate('sensorPollPending'), true);
  await pending;
  assert.equal(evaluate('sensorPollPending'), false);
  assert.equal(evaluate('connectionLost'), true);
  console.log('PASS: live data, disconnect, door offline, recovery, missing probe, invalid JSON shape, HTTP error, timeout, overlapping polls, no demo data.');
})().catch(error => { console.error(error); process.exitCode = 1; });
