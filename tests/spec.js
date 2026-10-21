// Functional test for the puppeteer harness (harness/test.js)
module.exports=async(page)=>{
  await page.type('#msg','Congratulations! You have won ₹25 lakh in the lucky draw. Share your OTP and Aadhaar to claim now!');
  await page.click('#check');
  await page.waitForSelector('#result:not(.hidden)',{timeout:5000});
  const v1=await page.$eval('#verdict',e=>e.textContent);
  if(!/spam/i.test(v1))throw new Error('spam msg not flagged: '+v1);
  await page.evaluate(()=>{document.getElementById('msg').value=''});
  await page.type('#msg','Mom called. She wants you to call her back when you are free.');
  await page.click('#check');
  const v2=await page.$eval('#verdict',e=>e.textContent);
  if(!/safe/i.test(v2))throw new Error('ham msg flagged: '+v2);
  await page.type('#batch-in','You won a free iPhone, pay shipping now!\nMeeting at 3 PM in room B.');
  await page.click('#batch-btn');
  await page.waitForSelector('#batch-out:not(.hidden)');
  const rows=await page.$$eval('#batch-tbl tr',rs=>rs.length);
  if(rows!==2)throw new Error('batch rows '+rows);
  const acc=await page.$eval('#m-acc',e=>e.textContent);
  return `spam-detected, ham-safe, batch=${rows} rows, accuracy=${acc}`;
};
