import { parquetMetadataAsync, parquetRead } from 'hyparquet';
import { fromPairs, sample, zip } from 'lodash';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';

import { DataType } from '../components/forms/ProjectCreationForm';

/**
 * loadParquetFile
 * parses a parquet file in memory
 * @param file: uploaded file
 * @returns a DataType object storing the data, fieldnames and original filename
 */
export async function loadParquetFile(file: File): Promise<DataType> {
  // since this function is async it must returns a Promise which contains a DataType object

  // read the uploaded file as an arrayBuffer
  const arrayBuffer = await file.arrayBuffer();
  // use th hyparquet library to read the file metadata (which stores the headers)
  const metadata = await parquetMetadataAsync(arrayBuffer);
  // here we have to create a new Promise as the parquetRead function only support the callback syntax
  // The Promise will be return immediately but it will be replaced by what is provided to its resolve function
  // It will be replaced only when the resolve function is called
  // we differ the resolution.
  return new Promise((resolve) =>
    parquetRead({
      metadata,
      file: arrayBuffer,
      // here is the callback which will be called once the file has been loaded
      onComplete: (arrayData) => {
        // extract headers from metadata
        const headers = metadata.schema.slice(1).map((s) => s.name);
        // transforme the data as an array of objects
        const data = arrayData.map((ad) => fromPairs(zip(headers, ad)));
        // resolve the promise with a dataType object
        resolve({ data, headers, filename: file.name });
      },
    }),
  );
}

/**
 * loadCSVFile
 * parses a csv file in memory
 * @param file: uploaded file
 * @returns a DataType object storing the data, fieldnames and original filename
 */
export async function loadCSVFile(file: File): Promise<DataType> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const csvContent = e.target?.result;

      if (typeof csvContent === 'string') {
        // detect separator: tab, semicolon, or comma
        const line = csvContent.split('\n')[0];
        const tabCount = (line.match(/\t/g) || []).length;
        const semicolonCount = (line.match(/;/g) || []).length;
        const commaCount = (line.match(/,/g) || []).length;
        const separator = tabCount > 3 ? '\t' : semicolonCount > commaCount ? ';' : ',';

        Papa.parse<Record<string, string>>(csvContent, {
          header: true,
          delimiter: separator,
          complete: (results) => {
            const headers = results.meta.fields || [];
            resolve({ data: results.data, headers, filename: file.name });
          },
        });
      } else {
        reject(new Error('Failed to read the CSV file as text.'));
      }
    };

    // Define the onerror callback for any file reading errors
    reader.onerror = () => {
      reject(new Error('Error reading the CSV file.'));
    };

    // Start reading the file as text
    reader.readAsText(file);
  });
}

export async function loadExcelFile(file: File): Promise<DataType> {
  // Read the uploaded file as an arrayBuffer
  const arrayBuffer = await file.arrayBuffer();

  // Use XLSX to read the file as a workbook
  // raw: false ensures values are formatted as strings (preserves large IDs)
  const workbook = XLSX.read(arrayBuffer, { type: 'array', raw: false });

  // Assuming we want the first sheet
  const sheetName = workbook.SheetNames[0];
  const worksheet = workbook.Sheets[sheetName];

  // Convert the sheet to JSON format with headers
  const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: '', raw: false });

  // Extract headers (first row), filtering out empty headers
  const rawHeaders = jsonData[0] as (string | null)[];
  const headers = rawHeaders.map((h, i) => (h ? String(h) : `__empty_${i}`)).filter(Boolean);

  // Extract data (subsequent rows) removing empty rows
  const data = jsonData
    .slice(1)
    .filter((row) => (row as string[]).some((cell) => cell !== null && cell !== ''))
    .map((row) => {
      const cells = row as (string | number | null)[];
      return fromPairs(headers.map((h, i) => [h, cells[i] != null ? String(cells[i]) : '']));
    });

  // Return a DataType object with the data, headers, and filename
  return { data, headers, filename: file.name };
}

// export async function loadFeatherFile(file: File): Promise<DataType> {
//   // Read the uploaded file as an arrayBuffer
//   const arrayBuffer = await file.arrayBuffer();

//   // Use the Apache Arrow library to read the Feather file
//   const table = tableFromIPC(new Uint8Array(arrayBuffer));

//   // Extract headers (column names)
//   const headers = table.schema.fields.map((field) => field.name);

//   // Extract the data as an array of objects
//   const data = [];
//   for (let i = 0; i < table.numRows; i++) {
//     const row = {};
//     headers.forEach((header, colIndex) => {
//       row[header] = table.getColumnAt(colIndex).get(i);
//     });
//     data.push(row);
//   }

//   // Resolve the promise with a DataType object
//   return { data, headers, filename: file.name };
// }

export async function loadFile(file: File): Promise<DataType | null> {
  let data;

  if (file.name.includes('parquet')) {
    console.log('parquet');
    try {
      data = await loadParquetFile(file);
    } catch (e) {
      console.error('Error reading parquet file:', e);
      return null;
    }
  } else if (file.name.includes('csv')) {
    console.log('csv');
    data = await loadCSVFile(file);
  } else if (file.name.includes('xlsx')) {
    console.log('xlsx');
    data = await loadExcelFile(file);
  } else {
    return null;
  }
  return data;
}

/**
 * Display the date to the "from ago" format.
 */
export function dateToFromAgo(date: Date): string {
  const seconds = Math.round((Date.now() - date.getTime()) / 1000);
  const prefix = seconds < 0 ? 'in ' : '';
  const suffix = seconds < 0 ? '' : ' ago';
  const absSecond = Math.abs(seconds);

  const times = [
    absSecond / 60 / 60 / 24 / 365, // years
    absSecond / 60 / 60 / 24 / 30, // months
    absSecond / 60 / 60 / 24 / 7, // weeks
    absSecond / 60 / 60 / 24, // days
    absSecond / 60 / 60, // hours
    absSecond / 60, // minutes
    absSecond, // seconds
  ];

  return (
    ['year', 'month', 'week', 'day', 'hour', 'minute', 'second']
      .map((name, index) => {
        const time = Math.floor(times[index]);
        if (time > 0) return `${prefix}${time} ${name}${time > 1 ? 's' : ''}${suffix}`;
        return null;
      })
      .reduce((acc, curr) => (acc === null && curr !== null ? curr : null), null) || 'now'
  );
}

// reorder function
export const reorderLabels = (labels: string[], orderedList: string[] | null): string[] => {
  if (!orderedList || orderedList.length === 0) {
    return labels;
  }
  const orderedSet = new Set(orderedList);
  const inOrder: string[] = [];
  const leftovers: string[] = [];
  for (const label of labels) {
    if (orderedSet.has(label)) {
      inOrder.push(label);
    } else {
      leftovers.push(label);
    }
  }
  inOrder.sort((a, b) => orderedList.indexOf(a) - orderedList.indexOf(b));
  return [...inOrder, ...leftovers];
};

export function getRandomName(prefix: string) {
  const now = new Date();
  let random_name: string = 'RDName';
  const monthName: string[] = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
  ];
  /*
  Names for QuickModel, BERTopic and bertmodel are sampled from the baby names 
  in england and Wales from 1996 to 2024 (25 boy names and 25 girl names)
  https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/livebirths/datasets/babynamesinenglandandwalesfrom1996
  */
  const animalsList = [
    'Albatross',
    'Alligator',
    'Alpaca',
    'Anaconda',
    'Ant',
    'Antelope',
    'Armadillo',
    'Baboon',
    'Badger',
    'Barracuda',
    'Basset',
    'Bat',
    'Beagle',
    'Bear',
    'Beauceron',
    'Beaver',
    'Bee',
    'Beetle',
    'Binturong',
    'Bison',
    'Boa',
    'Boar',
    'Bobcat',
    'Bouvier',
    'Boxer',
    'Briard',
    'Buffalo',
    'Bull',
    'Bulldog',
    'Bullmastiff',
    'Butterfly',
    'Camel',
    'Cardinal',
    'Cat',
    'Cheetah',
    'Chihuahua',
    'Chimp',
    'Chinchilla',
    'Chinook',
    'Chough',
    'Clownfish',
    'Cobra',
    'Cocker',
    'Cod',
    'Collie',
    'Condor',
    'Cormorant',
    'Cow',
    'Crab',
    'Crane',
    'Crocodile',
    'Crow',
    'Cuckoo',
    'Cur',
    'Dachshund',
    'Dalmatian',
    'Deer',
    'Dingo',
    'Dobermann',
    'Dolphin',
    'Donkey',
    'Dove',
    'Dragonfly',
    'Duck',
    'Dunker',
    'Dunlin',
    'Eagle',
    'Echidna',
    'Eel',
    'Elephant',
    'Elk',
    'Emu',
    'Falcon',
    'Ferret',
    'Finch',
    'Firefly',
    'Flamingo',
    'Fly',
    'Fox',
    'Foxhound',
    'Frog',
    'Galago',
    'Gecko',
    'Gerbil',
    'Gibbon',
    'Giraffe',
    'Gnat',
    'Gnu',
    'Goat',
    'Goldfinch',
    'Goose',
    'Gorilla',
    'Grasshopper',
    'Greyhound',
    'Griffon',
    'Grouse',
    'Guanaco',
    'Gull',
    'Hamster',
    'Hare',
    'Harrier',
    'Hawk',
    'Hedgehog',
    'Hen',
    'Heron',
    'Herring',
    'Hippo',
    'Hornet',
    'Horse',
    'Hummingbird',
    'Husky',
    'Hyena',
    'Iguana',
    'Jaguar',
    'Jay',
    'Jellyfish',
    'Kangaroo',
    'Kingbird',
    'Kinkajou',
    'Koala',
    'Labrador',
    'Ladybug',
    'Lapphund',
    'Lapwing',
    'Lark',
    'Lemur',
    'Leopard',
    'Lion',
    'Lizard',
    'Llama',
    'Lobster',
    'Locust',
    'Lupo',
    'Lynx',
    'Macaque',
    'Mandrill',
    'Marmoset',
    'Meerkat',
    'Mole',
    'Molossus',
    'Mongoose',
    'Moose',
    'Mosquito',
    'Moth',
    'Mouse',
    'Narwhal',
    'Newt',
    'Nightingale',
    'Numbat',
    'Ocelot',
    'Octopus',
    'Opossum',
    'Orangutan',
    'Ostritch',
    'Otter',
    'Owl',
    'Ox',
    'Oyster',
    'Panda',
    'Panther',
    'Parrot',
    'Partridge',
    'Peacock',
    'Peafowl',
    'Peccary',
    'Pelican',
    'Penguin',
    'Pheasant',
    'Pigeon',
    'Pinscher',
    'Pitbull',
    'Platypus',
    'Pony',
    'Poodle',
    'Porcupine',
    'Pug',
    'Python',
    'Quail',
    'Quokka',
    'Rabbit',
    'Raccoon',
    'Rajapalayam',
    'Ram',
    'Rat',
    'Rattlesnake',
    'Raven',
    'Reindeer',
    'Retriever',
    'Rhinoceros',
    'Rook',
    'Rooster',
    'Rottweiler',
    'Salamander',
    'Salmon',
    'Sandpiper',
    'Sardine',
    'Schnauzer',
    'Seahorse',
    'Seal',
    'Setter',
    'Shark',
    'Sheep',
    'Skunk',
    'Sloth',
    'Snail',
    'Spaniel',
    'Spider',
    'Squid',
    'Squirrel',
    'Starfish',
    'Starling',
    'Stingray',
    'Swallow',
    'Swan',
    'Tamarin',
    'Termite',
    'Terrier',
    'Thrush',
    'Tiger',
    'Toad',
    'Toucan',
    'Turkey',
    'Turtle',
    'Uguisu',
    'Unicorn',
    'Vicuna',
    'Viper',
    'Wallaby',
    'Wasp',
    'Weasel',
    'Whale',
    'Wildebeest',
    'Wolf',
    'Wombat',
    'Xenops',
    'Yak',
    'Zebra',
  ];
  if (prefix === 'BERTopic') {
    random_name = ('bertopic-' + sample(animalsList)) as string;
  }
  if (prefix === 'QuickModel') {
    random_name = ('quick-' + sample(animalsList)) as string;
  }
  if (prefix === 'feature') {
    const feature_names = ['Feature'];
    random_name = sample(feature_names) as string;
  }
  if (prefix === 'bertmodel') {
    random_name = ('bert-' + sample(animalsList)) as string;
  }
  if (prefix === 'Scheme') {
    random_name = 'New_Scheme';
  }
  if (prefix === 'Project') {
    random_name = 'Project';
  }
  const month = monthName[now.getMonth()];
  const day = String(now.getDate());
  const hour = ('0' + String(now.getHours())).slice(-2);
  const minute = ('0' + String(now.getMinutes())).slice(-2);
  // const year = String(now.getFullYear());

  return `${random_name}-${hour}h${minute}-${day}-${month}`;
}

export const sortDatesAsStrings = (
  timeA: string | undefined,
  timeB: string | undefined,
  reverse?: boolean,
) => {
  if (timeA && timeB) {
    const output = timeA < timeB ? -1 : timeA > timeB ? 1 : -1;
    if (reverse) {
      return -1 * output;
    }
    return output;
  }
  return 0;
};

export function isValidRegex(pattern: string) {
  console.log('isValid', pattern);
  try {
    new RegExp(pattern);
    return true;
  } catch (e) {
    console.log('pattern invalid');
    return false;
  }
}

export const displayTime = (time: string) => {
  // 2025-12-11 10:50:57.786644 -> 2025-12-11 10:50
  return time.slice(0, time.indexOf(':') + 3);
};

export function truncateInMiddle(string: string, maxLength: number, separator: string = '...') {
  if (string.length <= maxLength) return string;

  const sepLen = separator.length,
    charsToShow = maxLength - sepLen,
    frontChars = Math.ceil(charsToShow / 2),
    backChars = Math.floor(charsToShow / 2);

  return `${string.slice(0, frontChars)}${separator}${string.slice(string.length - backChars)}`;
}
